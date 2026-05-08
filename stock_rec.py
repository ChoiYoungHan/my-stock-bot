import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import html
import logging
from typing import Optional, Dict

# =========================================================
# 1. 로깅 설정
# =========================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# 필터 기준 완화 (사용자 요청 반영)
KOR_TOP_N = 500
USA_TOP_N = 500
MIN_SCORE = 20  # 40 -> 20으로 하향 (시그널 하나만 있어도 통과)

# =========================================================
# 시장 추세 필터
# =========================================================
def get_market_regime() -> Dict[str, bool]:
    regime = {"KOREA": True, "USA": True}
    try:
        indices = {"KOREA": "KS11", "USA": "SPY"}
        for mkt, ticker in indices.items():
            df = fdr.DataReader(ticker, (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'))
            if df.empty: continue
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            curr_price = df['Close'].iloc[-1]
            regime[mkt] = curr_price > ma20
    except Exception as e:
        logger.error(f"시장 추세 분석 실패: {e}")
    return regime

# =========================================================
# 지표 계산
# =========================================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    
    df["MA20"] = c.rolling(20).mean()
    df["BB_LOWER"] = df["MA20"] - (c.rolling(20).std() * 2)
    
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIG"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    df["Money"] = c * v
    df["MA5"] = c.rolling(5).mean()
    
    return df

# =========================================================
# 완화된 분석 로직
# =========================================================
def analyze_logic(ticker: str, df: pd.DataFrame, name: str, market: str, is_bull: bool) -> Optional[dict]:
    if len(df) < 50: return None
    df = calculate_indicators(df)
    curr, prev = df.iloc[-1], df.iloc[-2]
    
    # 2. 거래대금 필터 완화 (한국 50억, 미국 500만 달러 이상)
    min_money = 5_000_000_000 if market == "KOREA" else 5_000_000
    if curr["Money"] < min_money: return None

    # 8. 하락장 가중치 완화 (+20 -> +10)
    dynamic_min_score = MIN_SCORE if is_bull else MIN_SCORE + 10
    
    score = 0
    core_tags = []
    bonus_tags = []

    # 핵심 1: 볼린저 하단 (범위 1.01 -> 1.02로 소폭 확대)
    if curr["Low"] < curr["BB_LOWER"] * 1.02:
        score += 20; core_tags.append("BB_SUPP")
        
    # 핵심 2: RSI 과매도 (35 -> 40으로 범위 확대)
    if prev["RSI"] < 40 and curr["RSI"] > prev["RSI"]:
        score += 20; core_tags.append("RSI_REV")
        
    # 핵심 3: MACD 골든크로스
    if prev["MACD"] <= prev["MACD_SIG"] and curr["MACD"] > curr["MACD_SIG"]:
        score += 20; core_tags.append("MACD_GC")

    if score < dynamic_min_score: return None

    # 부가 조건
    body = abs(curr["Close"] - curr["Open"])
    total_range = curr["High"] - curr["Low"]
    if total_range > 0 and (body / total_range) < 0.2: # 도지 기준도 소폭 완화
        bonus_tags.append("DOJI")
    if curr["MA5"] > curr["MA20"]:
        bonus_tags.append("MA_UP")
    if curr["Close"] > curr["Open"] and (curr["High"] - curr["Close"]) < body * 0.2:
        bonus_tags.append("STRONG")

    return {
        "name": name, "score": score, "price": curr["Close"],
        "change": ((curr["Close"]/prev["Close"])-1)*100,
        "rsi": curr["RSI"], "core": core_tags, "bonus": bonus_tags
    }

# =========================================================
# 실행 및 전송 파트
# =========================================================
def send_telegram(msg: str):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try: requests.post(url, json=payload, timeout=20)
    except: pass

def process_market(market_name: str, tickers: list, names: dict, is_bull: bool):
    logger.info(f"[{market_name}] 분석 시작...")
    try:
        data = yf.download(tickers, period="12mo", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    except: return

    results = []
    for ticker in tickers:
        try:
            if ticker not in data.columns.levels[0]: continue
            df = data[ticker].dropna()
            if df.empty: continue
            res = analyze_logic(ticker, df, names.get(ticker, ticker), market_name, is_bull)
            if res: results.append(res)
        except: continue

    results = sorted(results, key=lambda x: -x["score"])[:10]
    if not results: return

    now = (datetime.utcnow() + timedelta(hours=9)).strftime("%y/%m/%d %H:%M")
    flag = "🇰🇷" if market_name == "KOREA" else "🇺🇸"
    msg = f"<b>{flag} {market_name} 바닥 반등 (기준완화)</b>\n"
    msg += f"<i>추세: {'🔵상승/횡보' if is_bull else '🔴하락(기준보정)'}</i>\n\n"
    
    for i, r in enumerate(results):
        core_str = " ".join([f"#{t}" for t in r["core"]])
        bonus_str = f" <code>[{', '.join(r['bonus'])}]</code>" if r["bonus"] else ""
        unit = "₩" if market_name == "KOREA" else "$"
        msg += f"{i+1}. <b>{html.escape(r['name'])}</b> ({r['score']}점){bonus_str}\n"
        msg += f"└ 💰 {unit}{r['price']:,.0f} ({r['change']:+.1f}%) | RSI:{r['rsi']:.0f}\n"
        msg += f"└ 📊 {core_str}\n\n"
    send_telegram(msg)

def main():
    regime = get_market_regime()
    try:
        kor = fdr.StockListing("KRX").sort_values("Marcap", ascending=False).head(KOR_TOP_N)
        kor_tickers = [str(c) + (".KS" if m == "KOSPI" else ".KQ") for c, m in zip(kor["Code"], kor["Market"])]
        process_market("KOREA", kor_tickers, dict(zip(kor_tickers, kor["Name"])), regime["KOREA"])
    except: pass

    try:
        us = fdr.StockListing("S&P500").head(USA_TOP_N)
        us_names = {str(row["Symbol"]).replace(".", "-"): row["Name"] for _, row in us.iterrows()}
        process_market("USA", list(us_names.keys()), us_names, regime["USA"])
    except: pass

if __name__ == "__main__":
    main()
