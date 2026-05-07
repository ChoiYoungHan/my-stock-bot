import os
import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import datetime
import requests
import logging

# yfinance 내부 로그 억제
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# --- [1. 환경 변수에서 설정 읽기] ---
# GitHub Secrets에 등록한 이름과 동일해야 합니다.
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(msg):
    if not msg: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=15)
    except: pass

def calculate_indicators(df):
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    df['MA5'] = c.rolling(5).mean()
    df['MA20'] = c.rolling(20).mean()
    df['MA60'] = c.rolling(60).mean()
    df['V_MA20'] = v.rolling(20).mean() # 거래량 20일 이평
    
    std = c.rolling(20).std()
    df['BB_U'], df['BB_L'] = df['MA20'] + (std * 2), df['MA20'] - (std * 2)
    
    delta = c.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0], down[down > 0] = 0, 0
    df['RSI'] = 100 - (100 / (1 + (up.ewm(13).mean() / down.abs().ewm(13).mean())))
    
    df['Tenkan'] = (h.rolling(9).max() + l.rolling(9).min()) / 2
    df['Kijun'] = (h.rolling(26).max() + l.rolling(26).min()) / 2
    return df

def analyze_logic(ticker, df, name):
    if len(df) < 60: return None, None
    df = calculate_indicators(df)
    curr, prev = df.iloc[-1], df.iloc[-2]
    
    # 거래량 필터 (평균 대비 1.2배 이상 터진 것만)
    if curr['Volume'] < curr['V_MA20'] * 1.2: return None, None

    res = {"name": name, "ticker": ticker, "price": curr['Close'], "change": ((curr['Close']/prev['Close'])-1)*100}
    
    # 1. 상승추세
    if curr['MA5'] > curr['MA20'] > curr['MA60'] and curr['Close'] > curr['Tenkan'] and curr['RSI'] > 50:
        return "TREND", res
    # 2. 바닥반등
    if prev['Close'] < prev['BB_L'] * 1.02 and curr['Close'] > curr['Open'] and curr['RSI'] < 50:
        return "REBOUND", res
    # 3. 골든크로스
    if (prev['MA5'] <= prev['MA20'] and curr['MA5'] > curr['MA20']) or (prev['Tenkan'] <= prev['Kijun'] and curr['Tenkan'] > curr['Kijun']):
        return "CROSS", res
            
    return None, None

def process_market(market_name, tickers, names):
    print(f"[{market_name}] 데이터 수집 및 분석 시작...")
    data = yf.download(tickers, period="8mo", group_by='ticker', threads=True, progress=False)
    
    storage = {"TREND": [], "REBOUND": [], "CROSS": []}
    for ticker in tickers:
        try:
            df = data[ticker].dropna()
            cat, res = analyze_logic(ticker, df, names[ticker])
            if cat: storage[cat].append(res)
        except: continue

    header = "🇰🇷" if market_name == "KOREA" else "🇺🇸"
    output = f"{header} **[{market_name} 시장 실시간 분석]**\n"
    output += f"분석 시간: {datetime.datetime.now().strftime('%H:%M')}\n\n"
    
    sections = [("TREND", "🚀 상승추세"), ("REBOUND", "⚓ 바닥반등"), ("CROSS", "✨ 골든크로스")]
    for key, title in sections:
        output += f"{title}\n"
        items = sorted(storage[key], key=lambda x: abs(x['change']), reverse=True)[:5]
        if not items:
            output += "└ 충족 종목 없음\n"
        for i in items:
            cur = "₩" if market_name == "KOREA" else "$"
            output += f"└ {i['name']} ({cur}{i['price']:,.2f}, {i['change']:+.2f}%)\n"
        output += "\n"
    
    send_telegram(output)

def main():
    # 1. 한국 시장
    kor_listing = fdr.StockListing('KRX').sort_values('Marcap', ascending=False).head(300)
    kor_tickers = [row['Code'] + (".KS" if row['Market'] == 'KOSPI' else ".KQ") for _, row in kor_listing.iterrows()]
    kor_names = dict(zip(kor_tickers, kor_listing['Name']))
    process_market("KOREA", kor_tickers, kor_names)

    # 2. 미국 시장
    us_listing = fdr.StockListing('S&P500')
    us_tickers = list(us_listing['Symbol'])
    us_names = dict(zip(us_listing['Symbol'], us_listing['Name']))
    process_market("USA", us_tickers, us_names)

if __name__ == "__main__":
    main()
