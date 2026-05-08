import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import html # 특수문자 처리를 위해 추가
from typing import Optional

# =========================================================
# 설정 및 환경변수
# =========================================================
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

KOR_TOP_N = 500
USA_TOP_N = 500
MIN_SCORE = 55

# =========================================================
# 텔레그램 전송 함수 (HTML 방식으로 변경하여 에러 방지)
# =========================================================
def send_telegram(msg: str):
    if not TOKEN or not CHAT_ID:
        print("⚠️ 전송 스킵: 설정이 없습니다.")
        return
    
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    # Markdown 대신 HTML 모드를 사용하여 파싱 에러를 원천 차단합니다.
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
    
    try:
        resp = requests.post(url, json=payload, timeout=20)
        if resp.status_code == 200:
            print("🚀 전송 성공!")
        else:
            print(f"❌ 전송 실패: {resp.text}")
    except Exception as e:
        print(f"❌ 에러: {e}")

# =========================================================
# 분석 로직 (기존 유지)
# =========================================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["MA5"] = c.rolling(5).mean()
    df["MA20"] = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["BB_LOWER"] = df["MA20"] - std20 * 2
    
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIG"] = df["MACD"].ewm(span=9).mean()
    
    low14, high14 = l.rolling(14).min(), h.rolling(14).max()
    df["STOCH_K"] = (c - low14) / (high14 - low14 + 1e-9) * 100
    df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()
    
    df["VMA20"] = v.rolling(20).mean()
    df["VOL_RATIO"] = v / (df["VMA20"] + 1e-9)
    return df

def analyze_logic(ticker: str, df: pd.DataFrame, name: str, market: str) -> Optional[dict]:
    if len(df) < 50: return None
    df = calculate_indicators(df)
    curr, prev = df.iloc[-1], df.iloc[-2]
    
    change = ((curr["Close"] / prev["Close"]) - 1) * 100
    if change >= 10.0: return None # 급등주 제외
    
    score, tags = 0, []
    if curr["Low"] < curr["BB_LOWER"] * 1.02:
        score += 15
        if curr["Close"] > curr["Open"]: score += 10; tags.append("BB_UP")
    if prev["RSI"] < 35 and curr["RSI"] > prev["RSI"]: score += 15; tags.append("RSI")
    if prev["STOCH_K"] < 20 and curr["STOCH_K"] > curr["STOCH_D"]: score += 15; tags.append("STOCH")
    if curr["MACD"] > curr["MACD_SIG"] and prev["MACD"] <= prev["MACD_SIG"]: score += 15; tags.append("MACD_GC")
    if curr["VOL_RATIO"] >= 1.5: score += 15; tags.append("VOL_UP")
    
    body, total = abs(curr["Close"] - curr["Open"]), curr["High"] - curr["Low"]
    if total > 0 and (body / total) < 0.15 and (total > body * 2): score += 15; tags.append("DOJI")

    if score < MIN_SCORE: return None
    return {"name": name, "score": score, "price": curr["Close"], "change": change, "rsi": curr["RSI"], "tags": tags[:3]}

# =========================================================
# 시장별 프로세스
# =========================================================
def process_market(market_name: str, tickers: list, names: dict):
    print(f"[{market_name}] 분석 시작...")
    try:
        data = yf.download(tickers, period="12mo", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    except: return

    results = []
    for ticker in tickers:
        try:
            df = data[ticker].dropna() if len(tickers) > 1 else data.dropna()
            if df.empty: continue
            res = analyze_logic(ticker, df, names.get(ticker, ticker), market_name)
            if res: results.append(res)
        except: continue

    results = sorted(results, key=lambda x: -x["score"])[:10]
    now = (datetime.utcnow() + timedelta(hours=9)).strftime("%y/%m/%d %H:%M")
    flag = "🇰🇷" if market_name == "KOREA" else "🇺🇸"
    
    if not results:
        print(f"[{market_name}] 조건 부합 종목 없음")
        return

    # HTML 태그를 사용하여 메시지 구성 (특수문자 안전 처리)
    msg = f"<b>{flag} {market_name} 바닥 반등 TOP 10</b> ({now})\n\n"
    for i, r in enumerate(results):
        safe_name = html.escape(r["name"]) # &, <, > 등 변환
        tag_str = " ".join([f"#{t}" for t in r["tags"]])
        unit = "₩" if market_name == "KOREA" else "$"
        msg += f"{i+1}. <b>{safe_name}</b> ({r['score']}점)\n"
        msg += f"└ 💰 {unit}{r['price']:,.0f} ({r['change']:+.1f}%) | RSI:{r['rsi']:.0f}\n"
        msg += f"└ 📊 {tag_str}\n\n"
    
    send_telegram(msg)

def main():
    # 1. 한국 (티커 필터링 강화)
    try:
        kor = fdr.StockListing("KRX").sort_values("Marcap", ascending=False).head(KOR_TOP_N)
        kor_tickers = [str(c) + (".KS" if m == "KOSPI" else ".KQ") for c, m in zip(kor["Code"], kor["Market"])]
        process_market("KOREA", kor_tickers, dict(zip(kor_tickers, kor["Name"])))
    except Exception as e: print(f"KOREA Error: {e}")

    # 2. 미국 (BRK.B -> BRK-B 변환 로직 정교화)
    try:
        us = fdr.StockListing("S&P500").head(USA_TOP_N)
        # Yahoo Finance는 BRK.B 대신 BRK-B를 사용함
        us_tickers = [str(t).replace(".", "-") for t in us["Symbol"]]
        process_market("USA", us_tickers, dict(zip(us_tickers, us["Name"])))
    except Exception as e: print(f"USA Error: {e}")

if __name__ == "__main__":
    main()
