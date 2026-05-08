import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

import requests
import logging
import os

# =========================================================
# 기본 설정
# =========================================================

logging.getLogger("yf").setLevel(logging.CRITICAL)

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

KOR_TOP_N = 500
USA_TOP_N = 500

MIN_SCORE = 55

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
        "text": msg,
        "parse_mode": "Markdown"
    }

    try:
        requests.post(url, json=payload, timeout=20)
    except Exception:
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

    # -----------------------------
    # 이동평균
    # -----------------------------

    df["MA5"] = c.rolling(5).mean()
    df["MA20"] = c.rolling(20).mean()
    df["MA60"] = c.rolling(60).mean()

    # -----------------------------
    # 볼린저 밴드
    # -----------------------------

    std20 = c.rolling(20).std()

    df["BB_UPPER"] = df["MA20"] + std20 * 2
    df["BB_LOWER"] = df["MA20"] - std20 * 2

    # -----------------------------
    # 거래량 평균
    # -----------------------------

    df["VMA20"] = v.rolling(20).mean()

    # -----------------------------
    # RSI
    # -----------------------------

    delta = c.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    # -----------------------------
    # MACD
    # -----------------------------

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # -----------------------------
    # 이격도
    # -----------------------------

    df["DISPARITY20"] = (c / df["MA20"]) * 100

    # -----------------------------
    # ADX
    # 추세 강도 측정
    # -----------------------------

    plus_dm = h.diff()
    minus_dm = l.diff() * -1

    plus_dm = np.where(
        (plus_dm > minus_dm) & (plus_dm > 0),
        plus_dm,
        0
    )

    minus_dm = np.where(
        (minus_dm > plus_dm) & (minus_dm > 0),
        minus_dm,
        0
    )

    tr1 = h - l
    tr2 = abs(h - c.shift())
    tr3 = abs(l - c.shift())

    tr = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(14).mean()

    plus_di = 100 * (
        pd.Series(plus_dm, index=df.index)
        .rolling(14)
        .mean() / atr
    )

    minus_di = 100 * (
        pd.Series(minus_dm, index=df.index)
        .rolling(14)
        .mean() / atr
    )

    dx = (
        abs(plus_di - minus_di)
        / (plus_di + minus_di)
    ) * 100

    df["ADX"] = dx.rolling(14).mean()

    # -----------------------------
    # 일목균형표 전환선
    # -----------------------------

    df["TENKAN"] = (
        h.rolling(9).max()
        + l.rolling(9).min()
    ) / 2

    return df

# =========================================================
# 캔들 분석
# =========================================================

def is_bullish_candle(row):

    return row["Close"] > row["Open"]

def candle_strength(row):

    body = abs(row["Close"] - row["Open"])

    total = row["High"] - row["Low"]

    if total == 0:
        return 0

    return body / total

# =========================================================
# 점수 기반 분석
# =========================================================

def analyze_logic(ticker, df, name):

    if len(df) < 120:
        return None

    df = calculate_indicators(df)

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    d2 = df.iloc[-3]

    score = 0
    reasons = []

    # =====================================================
    # 1. 바닥권 여부
    # =====================================================

    bb_near = (
        prev["Close"]
        <= prev["BB_LOWER"] * 1.04
    )

    if bb_near:
        score += 15
        reasons.append("BB하단")

    oversold = curr["DISPARITY20"] < 92

    if oversold:
        score += 10
        reasons.append("과매도")

    # =====================================================
    # 2. RSI 반등
    # =====================================================

    rsi_rebound = (
        prev["RSI"] < 35
        and curr["RSI"] > prev["RSI"]
    )

    if rsi_rebound:
        score += 15
        reasons.append("RSI반등")

    # =====================================================
    # 3. 거래량 증가
    # =====================================================

    volume_surge = (
        curr["Volume"]
        > curr["VMA20"] * 1.3
    )

    if volume_surge:
        score += 20
        reasons.append("거래량증가")

    # =====================================================
    # 4. MACD 골든크로스
    # =====================================================

    macd_cross = (
        prev["MACD"]
        <= prev["MACD_SIGNAL"]
        and curr["MACD"]
        > curr["MACD_SIGNAL"]
    )

    if macd_cross:
        score += 20
        reasons.append("MACD상승")

    # =====================================================
    # 5. 양봉
    # =====================================================

    bullish = is_bullish_candle(curr)

    if bullish:
        score += 10
        reasons.append("양봉")

    # =====================================================
    # 6. 장대양봉
    # =====================================================

    strong_candle = candle_strength(curr) > 0.6

    if strong_candle:
        score += 10
        reasons.append("장대양봉")

    # =====================================================
    # 7. MA 정배열
    # =====================================================

    trend = (
        curr["MA5"]
        > curr["MA20"]
        > curr["MA60"]
    )

    if trend:
        score += 15
        reasons.append("정배열")

    # =====================================================
    # 8. MA 골든크로스
    # =====================================================

    golden_cross = (
        prev["MA5"] <= prev["MA20"]
        and curr["MA5"] > curr["MA20"]
    )

    if golden_cross:
        score += 15
        reasons.append("골든크로스")

    # =====================================================
    # 9. ADX 상승
    # =====================================================

    adx_rising = (
        curr["ADX"] > prev["ADX"]
        and curr["ADX"] > 20
    )

    if adx_rising:
        score += 10
        reasons.append("추세강화")

    # =====================================================
    # 10. 일목 전환선 돌파
    # =====================================================

    tenkan_break = curr["Close"] > curr["TENKAN"]

    if tenkan_break:
        score += 10
        reasons.append("전환선돌파")

    # =====================================================
    # 점수별 등급
    # =====================================================

    if score >= 85:
        grade = "🔥 S급"
    elif score >= 70:
        grade = "⭐ A급"
    elif score >= 55:
        grade = "✅ B급"
    else:
        return None

    return {
        "ticker": ticker,
        "name": name,
        "score": score,
        "grade": grade,
        "price": curr["Close"],
        "change": (
            (curr["Close"] / prev["Close"]) - 1
        ) * 100,
        "rsi": curr["RSI"],
        "volume_ratio": (
            curr["Volume"] / curr["VMA20"]
        ),
        "reasons": ", ".join(reasons)
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

    except Exception as e:

        send_telegram(
            f"{market_name} 다운로드 실패\n{e}"
        )

        return

    candidates = []

    for ticker in tickers:

        try:

            df = data[ticker].dropna()

            result = analyze_logic(
                ticker,
                df,
                names[ticker]
            )

            if result:
                candidates.append(result)

        except Exception:
            continue

    # =====================================================
    # 점수 기준 정렬
    # =====================================================

    candidates = sorted(
        candidates,
        key=lambda x: (
            -x["score"],
            x["rsi"]
        )
    )

    now = datetime.utcnow() + timedelta(hours=9)

    flag = "🇰🇷" if market_name == "KOREA" else "🇺🇸"
    cur = "₩" if market_name == "KOREA" else "$"

    # =====================================================
    # 결과 없음
    # =====================================================

    if not candidates:

        send_telegram(
            f"{flag} [{market_name}] 조건 부합 종목 없음"
        )

        return

    # =====================================================
    # 메시지 생성
    # =====================================================

    msg = (
        f"{flag} *[{market_name} 바닥 반등 분석]*\n"
        f"{now.strftime('%Y-%m-%d %H:%M')}\n"
        f"===================="
    )

    # S/A/B 그룹

    groups = [
        "🔥 S급",
        "⭐ A급",
        "✅ B급"
    ]

    for grade in groups:

        filtered = [
            x for x in candidates
            if x["grade"] == grade
        ]

        if not filtered:
            continue

        msg += f"\n\n{grade}\n"

        for s in filtered[:10]:

            msg += (
                f"\n• {s['name']}"
                f"\n  점수: {s['score']}점"
                f"\n  가격: {cur}{s['price']:,.0f}"
                f"\n  등락: {s['change']:+.2f}%"
                f"\n  RSI: {s['rsi']:.1f}"
                f"\n  거래량: {s['volume_ratio']:.1f}배"
                f"\n  신호: {s['reasons']}\n"
            )

    send_telegram(msg)

# =========================================================
# 메인
# =========================================================

def main():

    # =====================================================
    # 한국 시장
    # =====================================================

    print("한국 시장 로딩")

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
    # 미국 시장
    # =====================================================

    print("미국 시장 로딩")

    us = fdr.StockListing("S&P500")

    us = us.head(USA_TOP_N)

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
