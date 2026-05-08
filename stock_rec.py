import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import logging
import os

logging.getLogger('yf').setLevel(logging.CRITICAL)

# --- [1. 설정 및 파라미터] ---
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(msg):
    if not (TOKEN and CHAT_ID):
        print("에러: 환경변수 미설정")
        return
    if not msg: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=25)
    except Exception as e:
        print(f"네트워크 에러: {e}")

def calculate_indicators(df):
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    df['MA5'] = c.rolling(5).mean()
    df['MA20'] = c.rolling(20).mean()
    df['MA60'] = c.rolling(60).mean()
    df['V_MA20'] = v.rolling(20).mean()
    std = c.rolling(20).std()
    df['BB_U'], df['BB_L'] = df['MA20'] + (std * 2), df['MA20'] - (std * 2)
    df['Max_60'] = h.shift(1).rolling(60).max()
    return df

def analyze_logic(ticker, df, name):
    if len(df) < 70: return None
    df = calculate_indicators(df)
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    d2 = df.iloc[-3]

    reasons = []
    priority = 99
    
    # 전략 분석 시작
    # 1. 박스권 돌파 (P1)
    if curr['Close'] > curr['Max_60'] and curr['Volume'] > curr['V_MA20'] * 1.5:
        priority = min(priority, 1)
        reasons.append(f"60일 신고가 돌파 및 거래량 폭증({curr['Volume']/curr['V_MA20']:.1f}배)")

    # 2. 도지 변곡 (P2)
    o, h, l, c = prev['Open'], prev['High'], prev['Low'], prev['Close']
    body = abs(c - o)
    is_true_doji = (body <= (h-l)*0.25) and (h-max(o,c) > body) and (min(o,c)-l > body) if (h-l)>0 else False
    if (d2['Close'] > prev['Close']) and is_true_doji and (curr['Close'] > curr['Open']) and (prev['Close'] <= prev['BB_L'] * 1.05):
        priority = min(priority, 2)
        tag = "양봉도지" if prev['Close'] > prev['Open'] else "음봉도지"
        reasons.append(f"하락 끝 {tag} 발생 후 당일 반등 성공")

    # 3. 정배열 초기 (P3)
    if not (prev['MA5'] > prev['MA20'] > prev['MA60']) and (curr['MA5'] > curr['MA20'] > curr['MA60']):
        priority = min(priority, 3)
        reasons.append("이동평균선(5/20/60) 정배열 초입 진입")

    # 4. 골든크로스 (P4)
    if (prev['MA5'] <= prev['MA20']) and (curr['MA5'] > curr['MA20']):
        priority = min(priority, 4)
        reasons.append("5일선이 20일선을 상향 돌파(골든크로스)")

    # 5. 바닥 반등 (P5)
    if prev['Close'] < prev['BB_L'] * 1.03 and curr['Close'] > curr['Open']:
        priority = min(priority, 5)
        reasons.append("불린저밴드 하단 지지 확인 및 양봉 전환")

    if not reasons: return None

    return {
        "name": name, 
        "price": curr['Close'], 
        "change": ((curr['Close']/prev['Close'])-1)*100,
        "priority": priority,
        "reason": " / ".join(reasons)
    }

def process_market(market_name, tickers, names):
    print(f"[{market_name}] 분석 중...")
    try:
        data = yf.download(tickers, period="8mo", group_by='ticker', threads=True, progress=False)
    except: return
    
    found_stocks = []
    for ticker in tickers:
        try:
            df = data[ticker].dropna()
            res = analyze_logic(ticker, df, names[ticker])
            if res: found_stocks.append(res)
        except: continue

    # 우선순위 -> 등락률 순으로 정렬
    found_stocks.sort(key=lambda x: (x['priority'], -abs(x['change'])))

    now_kst = datetime.utcnow() + timedelta(hours=9)
    header = "🇰🇷" if market_name == "KOREA" else "🇺🇸"
    cur_symbol = "₩" if market_name == "KOREA" else "$"

    if not found_stocks:
        send_telegram(f"{header} **[{market_name}]** 현재 조건에 부합하는 종목이 없습니다.")
        return

    # 메시지 작성 (분량 조절을 위해 상위 10개만 상세 출력)
    msg = f"{header} **[{market_name} 포착 리포트]**\n{now_kst.strftime('%m/%d %H:%M')}\n"
    msg += "*(우선순위 및 사유 포함)*\n\n"

    for i, s in enumerate(found_stocks[:10]):
        p_tag = f"P{s['priority']}"
        msg += f"**{i+1}. {s['name']}** ({cur_symbol}{s['price']:,.0f}, {s['change']:+.2f}%)\n"
        msg += f"┕ [등급: {p_tag}] {s['reason']}\n\n"

    send_telegram(msg)

def main():
    # 한국 500개
    kor_listing = fdr.StockListing('KRX').sort_values('Marcap', ascending=False).head(500)
    kor_tickers = [row['Code'] + (".KS" if row['Market'] == 'KOSPI' else ".KQ") for _, row in kor_listing.iterrows()]
    process_market("KOREA", kor_tickers, dict(zip(kor_tickers, kor_listing['Name'])))

    # 미국 S&P 500
    us_listing = fdr.StockListing('S&P500')
    us_tickers = [t.replace('.', '-') for t in us_listing['Symbol']]
    process_market("USA", us_tickers, dict(zip(us_tickers, us_listing['Name'])))

if __name__ == "__main__":
    main()
