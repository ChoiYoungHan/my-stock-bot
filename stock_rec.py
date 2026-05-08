import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import logging
import os

# yfinance 로그 억제
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
        requests.post(url, json=payload, timeout=20)
    except Exception as e:
        print(f"네트워크 에러: {e}")

def calculate_indicators(df):
    """기술적 지표 및 박스권 계산"""
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    df['MA5'] = c.rolling(5).mean()
    df['MA20'] = c.rolling(20).mean()
    df['MA60'] = c.rolling(60).mean()
    df['V_MA20'] = v.rolling(20).mean()
    
    # 볼린저 밴드
    std = c.rolling(20).std()
    df['BB_U'], df['BB_L'] = df['MA20'] + (std * 2), df['MA20'] - (std * 2)
    
    # 박스권 판단을 위한 60일 최고가 (당일 제외)
    df['Max_60'] = h.shift(1).rolling(60).max()
    
    return df

def analyze_logic(ticker, df, name):
    """추세 변곡, 돌파, 정배열 판별"""
    if len(df) < 70: return None, None
    df = calculate_indicators(df)
    
    try:
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        d2 = df.iloc[-3]
    except IndexError:
        return None, None

    res = {
        "name": name, "ticker": ticker, "price": curr['Close'], 
        "change": ((curr['Close']/prev['Close'])-1)*100
    }
    
    # --- [A. 장기 박스권 돌파] ---
    # 60일 최고가 돌파 + 거래량 실림(평균의 1.5배)
    if curr['Close'] > curr['Max_60'] and curr['Volume'] > curr['V_MA20'] * 1.5:
        return "BOX_BREAKOUT", res

    # --- [B. 정배열 초기 진입] ---
    # 전일에는 정배열이 아니었는데, 오늘 5 > 20 > 60 정배열 완성
    was_not_aligned = not (prev['MA5'] > prev['MA20'] > prev['MA60'])
    now_aligned = (curr['MA5'] > curr['MA20'] > curr['MA60'])
    if was_not_aligned and now_aligned:
        return "ALIGNED_START", res

    # --- [C. 도지 변곡점 로직] ---
    o, h, l, c = prev['Open'], prev['High'], prev['Low'], prev['Close']
    body = abs(c - o)
    is_true_doji = (body <= (h-l)*0.25) and (h-max(o,c) > body) and (min(o,c)-l > body) if (h-l)>0 else False
    
    if (d2['Close'] > prev['Close']) and is_true_doji and (curr['Close'] > curr['Open']) and (prev['Close'] <= prev['BB_L'] * 1.05):
        return "REVERSAL" if prev['Close'] > prev['Open'] else "ANY_DOJI", res

    # --- [D. 기타 기본 전략] ---
    if curr['Volume'] < curr['V_MA20'] * 0.4: return None, None
    
    if (prev['MA5'] <= prev['MA20']) and (curr['MA5'] > curr['MA20']): return "CROSS", res
    if prev['Close'] < prev['BB_L'] * 1.03 and curr['Close'] > curr['Open']: return "REBOUND", res
            
    return None, None

def process_market(market_name, tickers, names):
    print(f"[{market_name}] 분석 중...")
    try:
        data = yf.download(tickers, period="8mo", group_by='ticker', threads=True, progress=False)
    except: return
    
    storage = {k: [] for k in ["REVERSAL", "ANY_DOJI", "BOX_BREAKOUT", "ALIGNED_START", "CROSS", "REBOUND"]}
    
    for ticker in tickers:
        try:
            df = data[ticker].dropna() if len(tickers) > 1 else data.dropna()
            cat, res = analyze_logic(ticker, df, names[ticker])
            if cat: storage[cat].append(res)
        except: continue

    now_kst = datetime.utcnow() + timedelta(hours=9)
    header = "🇰🇷" if market_name == "KOREA" else "🇺🇸"
    cur_symbol = "₩" if market_name == "KOREA" else "$"

    # 1. 일반 리포트
    report = f"{header} **[{market_name} 리포트]**\n{now_kst.strftime('%m/%d %H:%M')}\n\n"
    for k, t in [("CROSS", "✨ 골든크로스"), ("REBOUND", "⚓ 바닥반등")]:
        report += f"{t}\n"
        items = sorted(storage[k], key=lambda x: abs(x['change']), reverse=True)[:5]
        report += ("\n".join([f"└ {i['name']} ({i['change']:+.2f}%)" for i in items]) if items else "└ 없음") + "\n\n"
    send_telegram(report)

    # 2. 강력 추세 리포트 (박스권 돌파 & 정배열 초기)
    trend_msg = f"{header} **[{market_name}] 강력 추세 시작**\n"
    found_trend = False
    for k, t in [("BOX_BREAKOUT", "🧨 박스권 돌파"), ("ALIGNED_START", "📈 정배열 초기")]:
        if storage[k]:
            found_trend = True
            trend_msg += f"\n{t}\n"
            for i in storage[k]:
                trend_msg += f"└ {i['name']} ({cur_symbol}{i['price']:,.0f})\n"
    if found_trend: send_telegram(trend_msg)

    # 3. 변곡점 리포트
    combined_rev = storage["REVERSAL"] + storage["ANY_DOJI"]
    if combined_rev:
        rev_msg = f"{header} **[{market_name}] 도지 변곡**\n"
        for i in combined_rev:
            rev_msg += f"└ {i['name']} ({cur_symbol}{i['price']:,.0f})\n"
        send_telegram(rev_msg)
    else:
        send_telegram(f"{header} [{market_name}] 도지 변곡 종목 없음")

def main():
    # 국장 500개
    kor_listing = fdr.StockListing('KRX').sort_values('Marcap', ascending=False).head(500)
    kor_tickers = [row['Code'] + (".KS" if row['Market'] == 'KOSPI' else ".KQ") for _, row in kor_listing.iterrows()]
    process_market("KOREA", kor_tickers, dict(zip(kor_tickers, kor_listing['Name'])))

    # 미장 S&P 500
    us_listing = fdr.StockListing('S&P500')
    us_tickers = [t.replace('.', '-') for t in us_listing['Symbol']]
    process_market("USA", us_tickers, dict(zip(us_tickers, us_listing['Name'])))

if __name__ == "__main__":
    main()
