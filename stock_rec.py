import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
import os

logging.getLogger('yf').setLevel(logging.CRITICAL)

# --- [설정] ---
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(msg):
    if not (TOKEN and CHAT_ID): return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=25)
    except: pass

def calculate_indicators(df):
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    # 이동평균선
    df['MA5'], df['MA20'], df['MA60'] = c.rolling(5).mean(), c.rolling(20).mean(), c.rolling(60).mean()
    df['V_MA20'] = v.rolling(20).mean()
    
    # 지표들
    tp = (h + l + c) / 3
    df['CCI'] = (tp - tp.rolling(14).mean()) / (0.015 * tp.rolling(14).apply(lambda x: np.fabs(x - x.mean()).mean()))
    
    delta = c.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0], down[down > 0] = 0, 0
    df['RSI'] = 100 - (100 / (1 + (up.ewm(13).mean() / down.abs().ewm(13).mean())))
    
    # 불린저 밴드 하단 (BB_L)
    std = c.rolling(20).std()
    df['BB_L'] = df['MA20'] - (std * 2)
    
    # 박스권/피보나치
    df['Max_60'] = h.shift(1).rolling(60).max()
    df['High_6m'] = h.rolling(120).max()
    df['Low_6m'] = l.rolling(120).min()
    
    return df

def analyze_logic(ticker, df, name):
    if len(df) < 120: return []
    df = calculate_indicators(df)
    
    curr, prev, d2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    
    # [공통 조건] 모든 전략은 기본적으로 불린저 밴드 하단 5% 이내에서 발생해야 함
    is_at_bottom = (prev['Close'] <= prev['BB_L'] * 1.05)
    
    if not is_at_bottom: return []
    
    matches = []

    # 1. 박스권 돌파 후 지지 (Retest) + BB 하단 지지
    recent_20 = df.iloc[-20:-1]
    if any(recent_20['Close'] > recent_20['Max_60']):
        support_line = curr['Max_60']
        if support_line * 0.98 <= curr['Close'] <= support_line * 1.03:
            matches.append("박스권 돌파 후 BB하단 지지")

    # 2. 피보나치 0.618 + BB 하단 반등
    fibo_618 = curr['High_6m'] - (curr['High_6m'] - curr['Low_6m']) * 0.618
    if prev['Close'] < fibo_618 and curr['Close'] > curr['Open']:
        matches.append("피보나치 0.618 & BB하단 반등")

    # 3. CCI 바닥 탈출 (과매도 중복 확인)
    if prev['CCI'] < -100 and curr['CCI'] > -100:
        matches.append("CCI & BB하단 과매도 탈출")

    # 4. 도지 변곡 (하락 끝자락)
    o, h, l, c = prev['Open'], prev['High'], prev['Low'], prev['Close']
    body = abs(c - o)
    is_true_doji = (body <= (h-l)*0.25) and (h-max(o,c) > body) and (min(o,c)-l > body) if (h-l)>0 else False
    if (d2['Close'] > prev['Close']) and is_true_doji and (curr['Close'] > curr['Open']):
        matches.append("BB하단 도지 캔들 변곡")

    # 5. 정배열 초입 (눌림목 정배열)
    if not (prev['MA5'] > prev['MA20'] > prev['MA60']) and (curr['MA5'] > curr['MA20'] > curr['MA60']):
        matches.append("BB하단 정배열 초입 진입")

    # 6. RSI 과매도 반등
    if prev['RSI'] < 35 and curr['Close'] > curr['Open']: # RSI 기준 소폭 완화
        matches.append("RSI & BB하단 기술적 반등")

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
    print(f"[{market_name}] 분석 중...")
    try: data = yf.download(tickers, period="10mo", group_by='ticker', threads=True, progress=False)
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
        send_telegram(f"{header} **[{market_name}]** BB하단 조건 부합 종목 없음")
        return

    msg = f"{header} **[{market_name} 바닥권 추세 분석]**\n{now.strftime('%m/%d %H:%M')}\n"
    msg += "*(모든 종목 불린저밴드 하단 위치)*\n"
    msg += "---"
    
    for cat, stocks in category_map.items():
        msg += f"\n\n🔵 **{cat}**\n"
        stocks.sort(key=lambda x: -abs(x['change']))
        for s in stocks[:8]:
            msg += f"└ {s['name']} ({cur_symbol}{s['price']:,.0f}, {s['change']:+.2f}%)\n"
    
    send_telegram(msg)

def main():
    # 국장 500개
    kor = fdr.StockListing('KRX').sort_values('Marcap', ascending=False).head(500)
    kor_t = [r['Code'] + (".KS" if r['Market'] == 'KOSPI' else ".KQ") for _, r in kor.iterrows()]
    process_market("KOREA", kor_t, dict(zip(kor_t, kor['Name'])))

    # 미장 S&P 500
    us = fdr.StockListing('S&P500')
    us_t = [t.replace('.', '-') for t in us['Symbol']]
    process_market("USA", us_t, dict(zip(us_t, us['Name'])))

if __name__ == "__main__":
    main()
