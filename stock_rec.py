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
    df['MA5'], df['MA20'], df['MA60'] = c.rolling(5).mean(), c.rolling(20).mean(), c.rolling(60).mean()
    df['V_MA20'] = v.rolling(20).mean()
    std = c.rolling(20).std()
    df['BB_L'] = df['MA20'] - (std * 2)
    
    # CCI 및 RSI
    tp = (h + l + c) / 3
    df['CCI'] = (tp - tp.rolling(14).mean()) / (0.015 * tp.rolling(14).apply(lambda x: np.fabs(x - x.mean()).mean()))
    delta = c.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0], down[down > 0] = 0, 0
    df['RSI'] = 100 - (100 / (1 + (up.ewm(13).mean() / down.abs().ewm(13).mean())))
    
    # 박스권 분석용 (60일 최고가)
    df['Max_60'] = h.shift(1).rolling(60).max()
    # 피보나치
    df['High_6m'] = h.rolling(120).max()
    df['Low_6m'] = l.rolling(120).min()
    
    return df

def analyze_logic(ticker, df, name):
    if len(df) < 120: return None
    df = calculate_indicators(df)
    
    curr, prev, d2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    reasons = []

    # 1. 박스권 돌파 후 지지 (BOX_RETEST)
    # 최근 20일 내 돌파 발생 여부 확인
    recent_20 = df.iloc[-20:-1]
    breakout_occured = any(recent_20['Close'] > recent_20['Max_60'])
    if breakout_occured:
        # 현재 주가가 과거 저항선(Max_60) 부근에 도달했는지 (-2% ~ +3% 오차)
        support_line = curr['Max_60']
        if support_line * 0.98 <= curr['Close'] <= support_line * 1.03:
            reasons.append("박스권 돌파 후 지지선 안착")

    # 2. 피보나치 0.618 반등
    fibo_618 = curr['High_6m'] - (curr['High_6m'] - curr['Low_6m']) * 0.618
    if prev['Close'] < fibo_618 and curr['Close'] > curr['Open']:
        reasons.append("피보나치 0.618 반등")

    # 3. CCI 과매도 탈출
    if prev['CCI'] < -100 and curr['CCI'] > -100:
        reasons.append("CCI 바닥 탈출")

    # 4. 도지 변곡
    o, h, l, c = prev['Open'], prev['High'], prev['Low'], prev['Close']
    body = abs(c - o)
    is_true_doji = (body <= (h-l)*0.25) and (h-max(o,c) > body) and (min(o,c)-l > body) if (h-l)>0 else False
    if (d2['Close'] > prev['Close']) and is_true_doji and (curr['Close'] > curr['Open']):
        reasons.append("도지 캔들 변곡")

    # 5. 정배열 초입
    if not (prev['MA5'] > prev['MA20'] > prev['MA60']) and (curr['MA5'] > curr['MA20'] > curr['MA60']):
        reasons.append("정배열 초입 진입")

    # 6. RSI 과매도 반등
    if prev['RSI'] < 30 and curr['Close'] > curr['Open']:
        reasons.append("RSI 과매도 반등")

    if not reasons: return None

    return {
        "name": name, "price": curr['Close'], "change": ((curr['Close']/prev['Close'])-1)*100,
        "reason": " / ".join(reasons)
    }

def process_market(market_name, tickers, names):
    print(f"[{market_name}] 분석 중...")
    try: data = yf.download(tickers, period="10mo", group_by='ticker', threads=True, progress=False)
    except: return
    
    found = []
    for t in tickers:
        try:
            df = data[t].dropna()
            res = analyze_logic(t, df, names[t])
            if res: found.append(res)
        except: continue

    found.sort(key=lambda x: -abs(x['change']))
    now = datetime.utcnow() + timedelta(hours=9)
    header, cur_symbol = ("🇰🇷", "₩") if market_name == "KOREA" else ("🇺🇸", "$")

    if not found:
        send_telegram(f"{header} **[{market_name}]** 현재 조건 부합 종목 없음")
        return

    msg = f"{header} **[{market_name} 추세 리포트]**\n{now.strftime('%m/%d %H:%M')}\n\n"
    for i, s in enumerate(found[:15]): # 최대 15개 출력
        msg += f"{i+1}. **{s['name']}** ({cur_symbol}{s['price']:,.0f}, {s['change']:+.2f}%) 🔵 _{s['reason']}_\n"
    
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
