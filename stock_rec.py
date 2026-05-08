import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
import os
from typing import Optional, List, Dict, Any  # 타입 힌트 호환성 추가

# =========================================================
# 설정
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
# 텔레그램
# =========================================================
def send_telegram(msg: str):
    if not (TOKEN and CHAT_ID):
        print(f"--- Telegram Config Missing ---\n{msg}")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg}
    try:
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"Telegram Error: {e}")

# =========================================================
# 지표 계산 (유틸리티 생략 - 기존과 동일)
# =========================================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # (사용자가 작성한 지표 계산 로직 그대로 유지)
    df = df.copy()
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    df["MA5"]  = c.rolling(5).mean()
    df["MA20"] = c.rolling(20).mean()
    df["MA60"] = c.rolling(60).mean()

    std20          = c.rolling(20).std()
    df["BB_UPPER"] = df["MA20"] + std20 * 2
    df["BB_LOWER"] = df["MA20"] - std20 * 2
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / (df["MA20"] + 1e-9)
    df["BB_WIDTH_MA60"] = df["BB_WIDTH"].rolling(60).mean()

    df["VMA20"]    = v.rolling(20).mean()
    df["VOL_RATIO"] = v / (df["VMA20"] + 1e-9)

    delta    = c.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12           = c.ewm(span=12, adjust=False).mean()
    ema26           = c.ewm(span=26, adjust=False).mean()
    df["MACD"]      = ema12 - ema26
    df["MACD_SIG"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIG"]

    df["DISPARITY20"] = (c / (df["MA20"] + 1e-9)) * 100

    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df["STOCH_K"] = (c - low14) / (high14 - low14 + 1e-9) * 100
    df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()
    df["WR"] = (high14 - c) / (high14 - low14 + 1e-9) * -100

    direction   = np.sign(c.diff()).fillna(0)
    df["OBV"]   = (v * direction).cumsum()
    df["OBV_MA5"] = df["OBV"].rolling(5).mean()

    tr1       = h - l
    tr2       = (h - c.shift(1)).abs()
    tr3       = (l - c.shift(1)).abs()
    tr        = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    return df

# ... (중간 캔들 유틸리티 함수 is_doji, is_hammer 등은 기존 코드 유지) ...
def candle_strength(row) -> float:
    body  = abs(row["Close"] - row["Open"])
    total = row["High"] - row["Low"]
    return 0.0 if total == 0 else body / total

def is_doji(row, threshold: float = 0.15) -> bool:
    body = abs(row["Close"] - row["Open"])
    total = row["High"] - row["Low"]
    return (body / total) < threshold if total != 0 else False

def is_hammer(row) -> bool:
    open_, close, high, low = row["Open"], row["Close"], row["High"], row["Low"]
    body = abs(close - open_) if abs(close - open_) != 0 else 1e-9
    lower_wick = min(open_, close) - low
    upper_wick = high - max(open_, close)
    return (lower_wick >= body * 2) and (upper_wick <= body * 0.5)

def is_inverted_hammer(row, prev_row) -> bool:
    open_, close, high, low = row["Open"], row["Close"], row["High"], row["Low"]
    body = abs(close - open_) if abs(close - open_) != 0 else 1e-9
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    return (upper_wick >= body * 2) and (lower_wick <= body * 0.5) and (prev_row["Close"] < prev_row["Open"])

def count_consecutive_down(df: pd.DataFrame, lookback: int = 10) -> int:
    count = 0
    closes, opens = df["Close"].values, df["Open"].values
    for i in range(len(df) - 2, max(len(df) - 2 - lookback, -1), -1):
        if closes[i] < opens[i]: count += 1
        else: break
    return count

def is_base_pattern(df: pd.DataFrame, window: int = 10, cv_thresh: float = 0.02) -> bool:
    recent_lows = df["Low"].iloc[-window:]
    mean_low = recent_lows.mean()
    if mean_low <= 0: return False
    return (recent_lows.std() / mean_low) < cv_thresh

# =========================================================
# 분석 로직 (TypeError 발생 지점 수정)
# =========================================================
def analyze_logic(ticker: str, df: pd.DataFrame, name: str, market: str) -> Optional[dict]:
    if len(df) < 120:
        return None

    try:
        df = calculate_indicators(df)
        curr = df.iloc[-1]
        prev = df.iloc[-2]
    except Exception:
        return None

    # 데이터 유효성 검사 (NaN 체크)
    if pd.isna(curr["Close"]) or pd.isna(prev["Close"]):
        return None

    change = ((curr["Close"] / prev["Close"]) - 1) * 100
    if change >= MAX_DAILY_CHANGE:
        return None

    vol_money = curr["Volume"] * curr["Close"]
    if market == "KOREA":
        if vol_money < 5_000_000_000: return None
    else:
        if vol_money < 5_000_000: return None

    score = 0
    tags = []

    # 지표 기반 점수 (사용자 로직 유지하되 안전장치 추가)
    if prev["Close"] <= prev["BB_LOWER"] * 1.04:
        score += 15; tags.append("BB")
    
    if curr["Low"] < curr.get("BB_LOWER", 0) and curr["Close"] > curr.get("MA20", 0):
        score += 20; tags.append("BB_REV")

    if pd.notna(curr.get("BB_WIDTH_MA60")) and curr["BB_WIDTH"] < curr["BB_WIDTH_MA60"] * 0.70:
        score += 10; tags.append("SQUEEZE")

    if prev["RSI"] < 35 and curr["RSI"] > prev["RSI"]:
        score += 20; tags.append("RSI")

    if prev["STOCH_K"] < 20 and curr["STOCH_K"] > curr["STOCH_D"] and prev["STOCH_K"] <= prev["STOCH_D"]:
        score += 15; tags.append("STOCH")

    if prev["WR"] < -80 and curr["WR"] > prev["WR"]:
        score += 10; tags.append("WR")

    if prev["MACD"] <= prev["MACD_SIG"] and curr["MACD"] > curr["MACD_SIG"]:
        score += 20; tags.append("MACD")

    if curr["VOL_RATIO"] >= 1.5:
        score += 20; tags.append("VOL")

    if is_hammer(curr):
        score += 15; tags.append("HAMMER")

    if ma_gc := (curr["MA5"] > curr["MA20"] and prev["MA5"] <= prev["MA20"]):
        score += 20; tags.append("GC")

    if score < MIN_SCORE:
        return None

    return {
        "name": name,
        "score": score,
        "price": curr["Close"],
        "change": change,
        "rsi": curr["RSI"],
        "stoch_k": curr["STOCH_K"],
        "wr": curr["WR"],
        "vol": curr["VOL_RATIO"],
        "tags": ",".join(tags),
    }

# =========================================================
# 시장 분석 및 실행부 (동일)
# =========================================================
def process_market(market_name: str, tickers: list, names: dict):
    print(f"[{market_name}] 분석 시작...")
    if not tickers: return

    try:
        # yfinance 멀티 다운로드 시 에러 방지를 위해 분할 처리 혹은 예외처리 강화
        data = yf.download(tickers, period="14mo", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    except Exception as e:
        print(f"Download Error: {e}")
        return

    results = []
    for ticker in tickers:
        try:
            df = data[ticker].dropna()
            if df.empty: continue
            res = analyze_logic(ticker, df, names.get(ticker, ticker), market_name)
            if res: results.append(res)
        except: continue

    results = sorted(results, key=lambda x: (-x["score"], x["rsi"]))
    
    # 메시지 생성 및 발송 (기존 로직과 동일)
    now = (datetime.utcnow() + timedelta(hours=9)).strftime("%m/%d %H:%M")
    flag = "[KR]" if market_name == "KOREA" else "[US]"
    
    if not results:
        send_telegram(f"{flag} {market_name}\n조건 부합 종목 없음")
        return

    msg = f"{flag} {market_name} 후보 ({now})\n\n"
    for r in results[:15]:
        msg += f"{r['name']} [{r['score']}]\n{r['price']:,.0f} ({r['change']:+.1f}%)\nRSI:{r['rsi']:.0f} V:{r['vol']:.1f}x\n{r['tags']}\n\n"
    
    send_telegram(msg)

def main():
    try:
        # 한국 시장
        kor = fdr.StockListing("KRX").sort_values("Marcap", ascending=False).head(KOR_TOP_N)
        kor_tickers = [row["Code"] + (".KS" if row["Market"] == "KOSPI" else ".KQ") for _, row in kor.iterrows()]
        process_market("KOREA", kor_tickers, dict(zip(kor_tickers, kor["Name"])))

        # 미국 시장
        us = fdr.StockListing("S&P500").head(USA_TOP_N)
        us_tickers = [t.replace(".", "-") for t in us["Symbol"]]
        process_market("USA", us_tickers, dict(zip(us_tickers, us["Name"])))
    except Exception as e:
        print(f"Main Error: {e}")

if __name__ == "__main__":
    main()
