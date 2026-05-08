import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

import requests
import logging
import os

# =========================================================
# 설정
# =========================================================

logging.getLogger("yf").setLevel(logging.CRITICAL)

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

KOR_TOP_N = 500
USA_TOP_N = 500

MIN_SCORE = 55

# 이미 급등한 종목 제외
MAX_DAILY_CHANGE = 10.0

# =========================================================
# 텔레그램
# =========================================================

def send_telegram(msg):

    if not (TOKEN and CHAT_ID):
        print(msg)
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": msg
    }

    try:
        requests.post(url, json=payload, timeout=20)
    except:
        pass

# =========================================================
# 지표 계산
# =========================================================

def calculate_indicators(df):

    df = df.copy()

    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    # 이동평균
    df["MA5"] = c.rolling(5).mean()
    df["MA20"] = c.rolling(20).mean()
    df["MA60"] = c.rolling(60).mean()

    # 볼린저
    std20 = c.rolling(20).std()

    df["BB_LOWER"] = df["MA20"] - std20 * 2

    # 거래량
    df["VMA20"] = v.rolling(20).mean()

    # RSI
    delta = c.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    # 이격도
    df["DISPARITY20"] = (
        c / df["MA20"]
    ) * 100

    # 거래량 추세
    df["VOL_RATIO"] = (
        df["Volume"] / df["VMA20"]
    )

    return df

# =========================================================
# 캔들 강도
# =========================================================

def candle_strength(row):

    body = abs(
        row["Close"] - row["Open"]
    )

    total = row["High"] - row["Low"]

    if total == 0:
        return 0

    return body / total

# =========================================================
# 분석 로직
# =========================================================

def analyze_logic(ticker, df, name):

    if len(df) < 120:
        return None

    df = calculate_indicators(df)

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # =====================================================
    # 이미 급등한 종목 제외
    # =====================================================

    change = (
        (curr["Close"] / prev["Close"]) - 1
    ) * 100

    if change >= MAX_DAILY_CHANGE:
        return None

    # =====================================================
    # 유동성 부족 제외
    # =====================================================

    avg_volume_money = (
        curr["Volume"] * curr["Close"]
    )

    if avg_volume_money < 5000000000:
        return None

    # =====================================================
    # 점수 계산
    # =====================================================

    score = 0
    tags = []

    # -----------------------------------------------------
    # 1. BB 하단
    # -----------------------------------------------------

    bb_near = (
        prev["Close"]
        <= prev["BB_LOWER"] * 1.04
    )

    if bb_near:
        score += 15
        tags.append("BB")

    # -----------------------------------------------------
    # 2. RSI 반등
    # -----------------------------------------------------

    rsi_rebound = (
        prev["RSI"] < 38
        and curr["RSI"] > prev["RSI"]
    )

    if rsi_rebound:
        score += 20
        tags.append("RSI")

    # -----------------------------------------------------
    # 3. 거래량 증가
    # -----------------------------------------------------

    volume_surge = (
        curr["VOL_RATIO"] >= 1.3
    )

    if volume_surge:
        score += 20
        tags.append("VOL")

    # -----------------------------------------------------
    # 4. MACD 상승 전환
    # -----------------------------------------------------

    macd_cross = (
        prev["MACD"]
        <= prev["MACD_SIGNAL"]
        and curr["MACD"]
        > curr["MACD_SIGNAL"]
    )

    if macd_cross:
        score += 20
        tags.append("MACD")

    # -----------------------------------------------------
    # 5. 양봉
    # -----------------------------------------------------

    bullish = (
        curr["Close"] > curr["Open"]
    )

    if bullish:
        score += 10

    # -----------------------------------------------------
    # 6. 장대양봉
    # -----------------------------------------------------

    if candle_strength(curr) > 0.6:
        score += 10
        tags.append("CANDLE")

    # -----------------------------------------------------
    # 7. 정배열 초기
    # -----------------------------------------------------

    ma_turn = (
        curr["MA5"] > curr["MA20"]
        and prev["MA5"] <= prev["MA20"]
    )

    if ma_turn:
        score += 20
        tags.append("GC")

    # -----------------------------------------------------
    # 8. 과매도
    # -----------------------------------------------------

    oversold = (
        curr["DISPARITY20"] < 94
    )

    if oversold:
        score += 10
        tags.append("OS")

    # =====================================================
    # 너무 오른 종목 제외
    # =====================================================

    # 최근 5일간 25% 이상 상승 제외

    recent_5d = (
        (curr["Close"] / df.iloc[-6]["Close"]) - 1
    ) * 100

    if recent_5d >= 25:
        return None

    # =====================================================
    # 최소 점수
    # =====================================================

    if score < MIN_SCORE:
        return None

    # =====================================================
    # 결과
    # =====================================================

    return {
        "name": name,
        "score": score,
        "price": curr["Close"],
        "change": change,
        "rsi": curr["RSI"],
        "vol": curr["VOL_RATIO"],
        "tags": ",".join(tags)
    }

# =========================================================
# 시장 분석
# =========================================================

def process_market(
    market_name,
    tickers,
    names
):

    print(f"[{market_name}] 분석 시작")

    try:

        data = yf.download(
            tickers,
            period="8mo",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False
        )

    except Exception:
        return

    results = []

    for ticker in tickers:

        try:

            df = data[ticker].dropna()

            result = analyze_logic(
                ticker,
                df,
                names[ticker]
            )

            if result:
                results.append(result)

        except:
            continue

    # =====================================================
    # 점수순 정렬
    # =====================================================

    results = sorted(
        results,
        key=lambda x: (
            -x["score"],
            x["rsi"]
        )
    )

    # =====================================================
    # 메시지 생성
    # =====================================================

    flag = (
        "🇰🇷"
        if market_name == "KOREA"
        else "🇺🇸"
    )

    cur = (
        "₩"
        if market_name == "KOREA"
        else "$"
    )

    now = (
        datetime.utcnow()
        + timedelta(hours=9)
    ).strftime("%m/%d %H:%M")

    # 결과 없음

    if not results:

        send_telegram(
            f"{flag} {market_name}\n조건 부합 종목 없음"
        )

        return

    # 간단한 형태

    msg = (
        f"{flag} {market_name} "
        f"바닥반등 후보 ({now})\n\n"
    )

    for r in results[:15]:

        msg += (
            f"{r['name']}  "
            f"[{r['score']}]\n"
            f"{cur}{r['price']:,.0f}  "
            f"{r['change']:+.1f}%  "
            f"RSI:{r['rsi']:.0f}  "
            f"V:{r['vol']:.1f}x\n"
            f"{r['tags']}\n\n"
        )

    send_telegram(msg)

# =========================================================
# 메인
# =========================================================

def main():

    # =====================================================
    # 한국
    # =====================================================

    kor = (
        fdr.StockListing("KRX")
        .sort_values(
            "Marcap",
            ascending=False
        )
        .head(KOR_TOP_N)
    )

    kor_tickers = []

    for _, row in kor.iterrows():

        suffix = (
            ".KS"
            if row["Market"] == "KOSPI"
            else ".KQ"
        )

        kor_tickers.append(
            row["Code"] + suffix
        )

    kor_names = dict(
        zip(
            kor_tickers,
            kor["Name"]
        )
    )

    process_market(
        "KOREA",
        kor_tickers,
        kor_names
    )

    # =====================================================
    # 미국
    # =====================================================

    us = (
        fdr.StockListing("S&P500")
        .head(USA_TOP_N)
    )

    us_tickers = [
        t.replace(".", "-")
        for t in us["Symbol"]
    ]

    us_names = dict(
        zip(
            us_tickers,
            us["Name"]
        )
    )

    process_market(
        "USA",
        us_tickers,
        us_names
    )

# =========================================================
# 실행
# =========================================================

if __name__ == "__main__":
    main()
