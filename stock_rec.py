import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
import os
from typing import Optional, List, Dict

# =========================================================
# 설정 및 환경변수
# =========================================================
logging.getLogger("yf").setLevel(logging.CRITICAL)

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

KOR_TOP_N = 500
USA_TOP_N = 500
MIN_SCORE = 55
MAX_DAILY_CHANGE = 10.0
MAX_5D_CHANGE = 25.0

# =========================================================
# 유틸리티 함수
# =========================================================
def send_telegram(msg: str):
    if not (TOKEN and CHAT_ID):
        print(msg)
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=20)
    except Exception as e:
        print(f"Telegram Error: {e}")

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # 이동평균 및 볼린저 밴드
    df["MA5"] = c.rolling(5).mean()
    df["MA20"] = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["BB_UPPER"] = df["MA20"] + std20 * 2
    df["BB_LOWER"] = df["MA20"] - std20 * 2
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / (df["MA20"] + 1e-9)
    df["BB_WIDTH_MA60"] = df["BB_WIDTH"].rolling(60).mean()

    # RSI, MACD, Stochastic
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIG"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIG"]

    low14, high14 = l.rolling(14).min(), h.rolling(14).max()
    df["STOCH_K"] = (c - low14) / (high14 - low14 + 1e-9) * 100
    df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()
    df["WR"] = (high14 - c) / (high14 - low14 + 1e-9) * -100

    # 거래량 및 이격도
    df["VMA20"] = v.rolling(20).mean()
    df["VOL_RATIO"] = v / (df["VMA20"] + 1e-9)
    df["DISPARITY20"] = (c / (df["MA20"] + 1e-9)) * 100

    return df

# =========================================================
# 캔들 패턴 로직 (보완)
# =========================================================
def is_doji(row, threshold: float = 0.15) -> bool:
    body = abs(row["Close"] - row["Open"])
    total = row["High"] - row["Low"]
    if total == 0: return False
    # 몸통이 짧고, 위아래 꼬리 합이 몸통보다 길어야 함
    return (body / total) < threshold and (total > body * 2)

def is_hammer(row) -> bool:
    o, c, h, l = row["Open"], row["Close"], row["High"], row["Low"]
    body = abs(c - o) or 1e-9
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    return (lower_wick >= body * 2) and (upper_wick <= body * 0.5)

# =========================================================
# 분석 메인 로직
# =========================================================
def analyze_logic(ticker: str, df: pd.DataFrame, name: str, market: str) -> Optional[dict]:
    if len(df) < 100: return None
    
    df = calculate_indicators(df)
    curr, prev = df.iloc[-1], df.iloc[-2]
    
    # 기본 필터: 급등주 제외 및 유동성 체크
    change = ((curr["Close"] / prev["Close"]) - 1) * 100
    if change >= MAX_DAILY_CHANGE: return None
    
    vol_money = curr["Volume"] * curr["Close"]
    money_limit = 5_000_000_000 if market == "KOREA" else 5_000_000
    if vol_money < money_limit: return None

    score = 0
    tags = []

    # 1. 볼린저 밴드 하단 전략 (가중치 강화)
    if curr["Low"] < curr["BB_LOWER"] * 1.02:
        score += 15
        if curr["Close"] > curr["Open"]: # 하단 근처 양봉
            score += 10
            tags.append("BB_UP")
        if is_doji(curr):
            score += 15
            tags.append("BB_DOJI")

    # 2. 보조지표 반등 (RSI, Stochastic, WR)
    if prev["RSI"] < 35 and curr["RSI"] > prev["RSI"]:
        score += 15; tags.append("RSI")
    if prev["STOCH_K"] < 20 and curr["STOCH_K"] > curr["STOCH_D"]:
        score += 15; tags.append("STOCH")
    if prev["WR"] < -80 and curr["WR"] > prev["WR"]:
        score += 10; tags.append("WR")

    # 3. 추세 및 거래량
    if curr["MACD"] > curr["MACD_SIG"] and prev["MACD"] <= prev["MACD_SIG"]:
        score += 15; tags.append("MACD_GC")
    if curr["VOL_RATIO"] >= 1.5:
        score += 15; tags.append("VOL_UP")
    if curr["MA5"] > curr["MA20"] and prev["MA5"] <= prev["MA20"]:
        score += 15; tags.append("MA_GC")
    
    if is_hammer(curr):
        score += 10; tags.append("HAMMER")

    if score < MIN_SCORE: return None

    return {
        "name": name, "score": score, "price": curr["Close"],
        "change": change, "rsi": curr["RSI"], "stoch_k": curr["STOCH_K"],
        "vol": curr["VOL_RATIO"], "tags": tags[:3] # 태그는 최대 3개만
    }

# =========================================================
# 실행 및 메시지 전송
# =========================================================
def process_market(market_name: str, tickers: list, names: dict):
    print(f"[{market_name}] 분석 중...")
    try:
        data = yf.download(tickers, period="12mo", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    except: return

    results = []
    for ticker in tickers:
        try:
            df = data[ticker].dropna()
            res = analyze_logic(ticker, df, names.get(ticker, ticker), market_name)
            if res: results.append(res)
        except: continue

    results = sorted(results, key=lambda x: -x["score"])[:10]
    
    if not results: return

    now = (datetime.utcnow() + timedelta(hours=9)).strftime("%y/%m/%d %H:%M")
    flag = "🇰🇷" if market_name == "KOREA" else "🇺🇸"
    msg = f"{flag} *{market_name} 바닥 반등 TOP 10* ({now})\n\n"
    
    for i, r in enumerate(results):
        tag_str = " ".join([f"#{t}" for t in r["tags"]])
        msg += f"{i+1}. *{r['name']}*  ({r['score']}점)\n"
        msg += f"└ 💰 {r['price']:,.0f} ({r['change']:+.1f}%) | RSI:{r['rsi']:.0f}\n"
        msg += f"└ 📊 {tag_str}\n\n"

    send_telegram(msg)

def main():
    # 한국 시장 (KOSPI/KOSDAQ 시총 상위)
    try:
        kor = fdr.StockListing("KRX").sort_values("Marcap", ascending=False).head(KOR_TOP_N)
        kor_tickers = [c + (".KS" if m == "KOSPI" else ".KQ") for c, m in zip(kor["Code"], kor["Market"])]
        process_market("KOREA", kor_tickers, dict(zip(kor_tickers, kor["Name"])))

        # 미국 시장 (S&P500 상위)
        us = fdr.StockListing("S&P500").head(USA_TOP_N)
        us_tickers = [t.replace(".", "-") for t in us["Symbol"]]
        process_market("USA", us_tickers, dict(zip(us_tickers, us["Name"])))
    except Exception as e:
        print(f"Main Error: {e}")

if __name__ == "__main__":
    main()
