import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
import os

logging.getLogger('yf').setLevel(logging.CRITICAL)

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(msg):
    if not (TOKEN and CHAT_ID):
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown"
    }

    try:
        requests.post(url, json=payload, timeout=20)
    except:
        pass

def calculate_indicators(df):

    c = df['Close']
    h = df['High']
    l = df['Low']
    v = df['Volume']

    # 이동평균
    df['MA5'] = c.rolling(5).mean()
    df['MA20'] = c.rolling(20).mean()
    df['MA60'] = c.rolling(60).mean()

    # 볼린저
    std20 = c.rolling(20).std()
    df['BB_U'] = df['MA20'] + std20 * 2
    df['BB_L'] = df['MA20'] - std20 * 2

    # 거래량
    df['VMA20'] = v.rolling(20).mean()

    # RSI
    delta = c.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()

    df['MACD'] = ema12 - ema26
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 이격도
    df['Disparity20'] = (c / df['MA20']) * 100

    # 일목
    df['Tenkan'] = (h.rolling(9).max() + l.rolling(9).min()) / 2

    return df

def analyze_logic(ticker, df, name):

    if len(df) < 80:
        return []

    df = calculate_indicators(df)

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    d2 = df.iloc[-3]

    matches = []

    # -----------------------------
    # 1. 상승추세
    # -----------------------------
    if (
        curr['MA5'] > curr['MA20'] > curr['MA60']
        and curr['Close'] > curr['Tenkan']
    ):
        matches.append("상승추세")

    # -----------------------------
    # 2. 실전형 바닥 반등
    # -----------------------------

    bb_near = prev['Close'] <= prev['BB_L'] * 1.03

    rsi_rebound = (
        prev['RSI'] < 35
        and curr['RSI'] > prev['RSI']
    )

    volume_surge = (
        curr['Volume'] > curr['VMA20'] * 1.5
    )

    macd_cross = (
        prev['MACD'] <= prev['MACD_SIGNAL']
        and curr['MACD'] > curr['MACD_SIGNAL']
    )

    oversold = curr['Disparity20'] < 92

    bullish_candle = curr['Close'] > curr['Open']

    if (
        bb_near
        and rsi_rebound
        and volume_surge
        and macd_cross
        and oversold
        and bullish_candle
    ):
        matches.append("강한 바닥반등")

    # -----------------------------
    # 3. 골든크로스
    # -----------------------------
    if (
        prev['MA5'] <= prev['MA20']
        and curr['MA5'] > curr['MA20']
    ):
        matches.append("골든크로스")

    results = []

    for m in matches:

        results.append({
            "category": m,
            "name": name,
            "price": curr['Close'],
            "change": ((curr['Close'] / prev['Close']) - 1) * 100,
            "rsi": curr['RSI']
        })

    return results

def process_market(market_name, tickers, names):

    print(f"[{market_name}] 분석 시작...")

    try:
        data = yf.download(
            tickers,
            period="6mo",
            group_by='ticker',
            threads=True,
            progress=False
        )
    except:
        return

    category_map = {}

    for t in tickers:

        try:
            df = data[t].dropna()

            res_list = analyze_logic(t, df, names[t])

            for res in res_list:

                cat = res['category']

                if cat not in category_map:
                    category_map[cat] = []

                category_map[cat].append(res)

        except:
            continue

    now = datetime.utcnow() + timedelta(hours=9)

    header, cur_symbol = (
        ("🇰🇷", "₩")
        if market_name == "KOREA"
        else ("🇺🇸", "$")
    )

    if not category_map:

        send_telegram(
            f"{header} [{market_name}] 조건 부합 종목 없음"
        )

        return

    msg = (
        f"{header} [{market_name} 시장 분석]\n"
        f"{now.strftime('%m/%d %H:%M')}\n"
        f"-------------------"
    )

    categories = [
        "강한 바닥반등",
        "골든크로스",
        "상승추세"
    ]

    for cat in categories:

        if cat in category_map:

            msg += f"\n\n📌 {cat}\n"

            stocks = sorted(
                category_map[cat],
                key=lambda x: -abs(x['change'])
            )

            for s in stocks[:10]:

                msg += (
                    f"└ {s['name']} "
                    f"({cur_symbol}{s['price']:,.0f}, "
                    f"{s['change']:+.2f}%, "
                    f"RSI:{s['rsi']:.1f})\n"
                )

    send_telegram(msg)

def main():

    # 한국
    kor = (
        fdr.StockListing('KRX')
        .sort_values('Marcap', ascending=False)
        .head(500)
    )

    kor_t = [
        r['Code'] + (
            ".KS"
            if r['Market'] == 'KOSPI'
            else ".KQ"
        )
        for _, r in kor.iterrows()
    ]

    process_market(
        "KOREA",
        kor_t,
        dict(zip(kor_t, kor['Name']))
    )

    # 미국
    us = fdr.StockListing('S&P500')

    us_t = [
        t.replace('.', '-')
        for t in us['Symbol']
    ]

    process_market(
        "USA",
        us_t,
        dict(zip(us_t, us['Name']))
    )

if __name__ == "__main__":
    main()
