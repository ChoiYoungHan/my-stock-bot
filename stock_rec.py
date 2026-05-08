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
    if not (TOKEN and CHAT_ID): return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=20)
    except: pass

def calculate_indicators(df):
    c, h, l = df['Close'], df['High'], df['Low']
    # 이동평균선
    df['MA5'] = c.rolling(5).mean()
    df['MA20'] = c.rolling(20).mean()
    df['MA60'] = c.rolling(60).mean()
    # 볼린저 밴드 하단
    df['BB_L'] = df['MA20'] - (c.rolling(20).std() * 2)
    # 일목균형표 전환선
    df['Tenkan'] = (h.rolling(9).max() + l.rolling(9).min()) / 2
    return df

def analyze_logic(ticker, df, name):
    if len(df) < 30: return []
    df = calculate_indicators(df)
    
    curr, prev, d2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    matches = []

    # 1. 상승추세 (TREND)
    if curr['MA5'] > curr['MA20'] > curr['MA60'] and curr['Close'] > curr['Tenkan']:
        matches.append("상승추세 (정배열)")

    # 2. 바닥반등 (REBOUND)
    if prev['Close'] < prev['BB_L'] * 1.03 and curr['Close'] > curr['Open']:
        matches.append("바닥반등 (BB하단 지지)")

    # 3. 골든크로스 (CROSS)
    if prev['MA5'] <= prev['MA20'] and curr['MA5'] > curr['MA20']:
        matches.append("골든크로스 (5/20)")

    # 4. 변곡점 (REVERSAL) - 하락 후 양봉 도지 발생
    o, h, l, c = prev['Open'], prev['High'], prev['Low'], prev['Close']
    body = abs(c - o)
    is_positive_doji = (c > o) and (body <= (h-l)*0.25) if (h-l)>0 else False
    if (d2['Close'] > prev['Close']) and is_positive_doji and (curr['Close'] > curr['Open']):
        matches.append("변곡점 (양봉도지 반등)")

    results = []
    for m in matches:
        results.append({
            "category": m,
            "name": name,
            "price": curr['Close'],
            "change": ((curr['Close']/prev['Close'])-1)*100
        })
    return results

def process_market(market_name, tickers, names):
    print(f"[{market_name}] 분석 시작...")
    try: data = yf.download(tickers, period="6mo", group_by='ticker', threads=True, progress=False)
    except: return
    
    category_map = {}
    for t in tickers:
        try:
            df = data[t].dropna()
            res_list = analyze_logic(t, df, names[t])
            for res in res_list:
                cat = res['category']
                if cat not in category_map: category_map[cat] = []
                category_map[cat].append(res)
        except: continue

    now = datetime.utcnow() + timedelta(hours=9)
    header, cur_symbol = ("🇰🇷", "₩") if market_name == "KOREA" else ("🇺🇸", "$")

    if not category_map:
        send_telegram(f"{header} **[{market_name}]** 포착 종목 없음")
        return

    msg = f"{header} **[{market_name} 시장 분석]**\n{now.strftime('%m/%d %H:%M')}\n---"
    
    # 사유별로 그룹화하여 출력
    for cat in ["상승추세 (정배열)", "바닥반등 (BB하단 지지)", "골든크로스 (5/20)", "변곡점 (양봉도지 반등)"]:
        if cat in category_map:
            msg += f"\n\n🔵 **{cat}**\n"
            stocks = sorted(category_map[cat], key=lambda x: -abs(x['change']))
            for s in stocks[:8]:
                msg += f"└ {s['name']} ({cur_symbol}{s['price']:,.0f}, {s['change']:+.2f}%)\n"
    
    send_telegram(msg)

def main():
    # 한국 상위 500개
    kor = fdr.StockListing('KRX').sort_values('Marcap', ascending=False).head(500)
    kor_t = [r['Code'] + (".KS" if r['Market'] == 'KOSPI' else ".KQ") for _, r in kor.iterrows()]
    process_market("KOREA", kor_t, dict(zip(kor_t, kor['Name'])))

    # 미국 S&P 500
    us = fdr.StockListing('S&P500')
    us_t = [t.replace('.', '-') for t in us['Symbol']]
    process_market("USA", us_t, dict(zip(us_t, us['Name'])))

if __name__ == "__main__":
    main()
