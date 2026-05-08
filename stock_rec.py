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
        print("에러: 환경변수(TELEGRAM_TOKEN, CHAT_ID)를 설정해주세요.")
        return
    if not msg: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=20)
    except Exception as e:
        print(f"네트워크 에러: {e}")

def calculate_indicators(df):
    """기술적 지표 계산"""
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    df['MA5'] = c.rolling(5).mean()
    df['MA20'] = c.rolling(20).mean()
    df['MA60'] = c.rolling(60).mean()
    df['V_MA20'] = v.rolling(20).mean()
    
    # 불린저 밴드
    std = c.rolling(20).std()
    df['BB_U'], df['BB_L'] = df['MA20'] + (std * 2), df['MA20'] - (std * 2)
    
    # 일목균형표 전환선
    df['Tenkan'] = (h.rolling(9).max() + l.rolling(9).min()) / 2
    return df

def analyze_logic(ticker, df, name):
    """추세 변곡 및 매매 전략 판별 (조건 완화 버전)"""
    if len(df) < 30: return None, None
    df = calculate_indicators(df)
    
    try:
        curr = df.iloc[-1]    # 당일
        prev = df.iloc[-2]    # 전일 (양봉 도지 기대)
        d2 = df.iloc[-3]      # 2일 전 (하락)
    except IndexError:
        return None, None

    res = {
        "name": name, "ticker": ticker, "price": curr['Close'], 
        "change": ((curr['Close']/prev['Close'])-1)*100
    }
    
    # --- [전략 1: 추세 변곡(REVERSAL) - 조건 완화형] ---
    # 1. 하락 흐름: 2일 전보다 전일 종가가 낮음
    is_prev_falling = d2['Close'] > prev['Close']
    
    # 2. 변곡점(전일): '양봉 도지' (몸통이 시가보다 크고, 전체 변동폭의 25% 이내)
    prev_total = (prev['High'] - prev['Low'])
    is_positive_doji = (prev['Close'] > prev['Open']) and \
                       ((prev['Close'] - prev['Open']) <= prev_total * 0.25 if prev_total > 0 else False)
    
    # 3. 위치: 전일 시가가 2일 전 종가보다 낮음 (하락 관성 유지)
    is_low_start = prev['Open'] < d2['Close']
    
    # 4. 반등확인(당일): 양봉 형성
    is_today_bullish = curr['Close'] > curr['Open']
    
    # 5. 과매도 구간: 불린저 밴드 하단 5% 이내 (완화됨)
    is_near_bottom = (prev['Close'] <= prev['BB_L'] * 1.05)

    if is_prev_falling and is_positive_doji and is_low_start and is_today_bullish and is_near_bottom:
        return "REVERSAL", res

    # --- 공통 필터 ---
    if curr['Volume'] < curr['V_MA20'] * 0.4: # 거래량 필터도 살짝 완화 (0.5 -> 0.4)
        return None, None

    # --- [전략 2: 골든크로스 (5일/20일)] ---
    if (df['MA5'].iloc[-2] <= df['MA20'].iloc[-2]) and (df['MA5'].iloc[-1] > df['MA20'].iloc[-1]):
        return "CROSS", res

    # --- [전략 3: 상승추세 (정배열)] ---
    if curr['MA5'] > curr['MA20'] > curr['MA60'] and curr['Close'] > curr['Tenkan']:
        return "TREND", res

    # --- [전략 4: 바닥 반등] ---
    if prev['Close'] < prev['BB_L'] * 1.03 and curr['Close'] > curr['Open']:
        return "REBOUND", res
            
    return None, None

def process_market(market_name, tickers, names):
    print(f"[{market_name}] 분석 시작...")
    try:
        data = yf.download(tickers, period="6mo", group_by='ticker', threads=True, progress=False)
    except Exception as e:
        print(f"다운로드 실패: {e}")
        return
    
    storage = {"TREND": [], "REBOUND": [], "CROSS": [], "REVERSAL": []}
    for ticker in tickers:
        try:
            df = data[ticker].dropna() if len(tickers) > 1 else data.dropna()
            if df.empty or len(df) < 15: continue
            cat, res = analyze_logic(ticker, df, names[ticker])
            if cat: storage[cat].append(res)
        except: continue

    now_kst = datetime.utcnow() + timedelta(hours=9)
    header = "🇰🇷" if market_name == "KOREA" else "🇺🇸"
    cur_symbol = "₩" if market_name == "KOREA" else "$"
    
    # 1. 일반 리포트 전송
    output = f"{header} **[{market_name} 시장 리포트]**\n분석: {now_kst.strftime('%m/%d %H:%M')}\n\n"
    has_common_data = False
    
    for key, title in [("TREND", "🚀 상승추세"), ("REBOUND", "⚓ 바닥반등"), ("CROSS", "✨ 골든크로스")]:
        output += f"{title}\n"
        items = sorted(storage[key], key=lambda x: abs(x['change']), reverse=True)[:5]
        if items:
            has_common_data = True
            for i in items:
                output += f"└ {i['name']} ({cur_symbol}{i['price']:,.0f}, {i['change']:+.2f}%)\n"
        else:
            output += "└ 충족 종목 없음\n"
        output += "\n"
    
    send_telegram(output)

    # 2. 추세 변곡(REVERSAL) 상세 리포트
    if storage["REVERSAL"]:
        rev_msg = f"{header} **[{market_name}] 완화된 추세변곡 시그널**\n"
        rev_msg += "*(하락 후 양봉도지 지지 확인)*\n\n"
        for i in storage["REVERSAL"]:
            rev_msg += f"💎 **{i['name']}**\n"
            rev_msg += f"└ 현재가: {cur_symbol}{i['price']:,.0f} ({i['change']:+.2f}%)\n\n"
        send_telegram(rev_msg)
    else:
        # 종목이 없을 경우 메시지 전송
        send_telegram(f"{header} **[{market_name}]** 현재 조건(REVERSAL)을 만족하는 종목이 없습니다.")

def main():
    # 1. 한국 시장 (시총 상위 300)
    try:
        kor_listing = fdr.StockListing('KRX').sort_values('Marcap', ascending=False).head(300)
        kor_tickers = [row['Code'] + (".KS" if row['Market'] == 'KOSPI' else ".KQ") for _, row in kor_listing.iterrows()]
        process_market("KOREA", kor_tickers, dict(zip(kor_tickers, kor_listing['Name'])))
    except Exception as e:
        print(f"한국 시장 분석 중 오류: {e}")

    # 2. 미국 시장 (S&P 500)
    try:
        us_listing = fdr.StockListing('S&P500')
        us_tickers = [t.replace('.', '-') for t in us_listing['Symbol']]
        process_market("USA", us_tickers, dict(zip(us_tickers, us_listing['Name'])))
    except Exception as e:
        print(f"미국 시장 분석 중 오류: {e}")

if __name__ == "__main__":
    main()
