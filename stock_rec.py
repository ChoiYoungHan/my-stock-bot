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
# 1. 로깅 및 환경 설정
# =========================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

KOR_TOP_N = 500
USA_TOP_N = 500
MIN_SCORE = 20 

# =========================================================
# 2. 시장 추세 및 지표 계산
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

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty: return df
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    
    df["MA20"] = c.rolling(20).mean()
    df["STD"] = c.rolling(20).std()
    df["BB_LOWER"] = df["MA20"] - (df["STD"] * 2)
    
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIG"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    df["Money"] = c * v
    df["MA5"] = c.rolling(5).mean()
    
    return df

# =========================================================
# 3. 분석 로직
# =========================================================

# (1) 기존 일봉 스코어링 로직
def analyze_logic(ticker: str, df: pd.DataFrame, name: str, market: str, is_bull: bool) -> Optional[dict]:
    if len(df) < 50: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    
    min_money = 5_000_000_000 if market == "KOREA" else 5_000_000
    if curr["Money"] < min_money: return None

    dynamic_min_score = MIN_SCORE if is_bull else MIN_SCORE + 10
    score = 0
    core_tags = []
    bonus_tags = []

    if curr["Low"] < curr["BB_LOWER"] * 1.02:
        score += 20; core_tags.append("BB_SUPP")
    if prev["RSI"] < 40 and curr["RSI"] > prev["RSI"]:
        score += 20; core_tags.append("RSI_REV")
    if prev["MACD"] <= prev["MACD_SIG"] and curr["MACD"] > curr["MACD_SIG"]:
        score += 20; core_tags.append("MACD_GC")

    if score < dynamic_min_score: return None

    body = abs(curr["Close"] - curr["Open"])
    total_range = curr["High"] - curr["Low"]
    if total_range > 0 and (body / total_range) < 0.2: bonus_tags.append("DOJI")
    if curr["MA5"] > curr["MA20"]: bonus_tags.append("MA_UP")
    if curr["Close"] > curr["Open"] and (curr["High"] - curr["Close"]) < body * 0.2: bonus_tags.append("STRONG")

    return {
        "name": name, "score": score, "price": curr["Close"],
        "change": ((curr["Close"]/prev["Close"])-1)*100,
        "rsi": curr["RSI"], "core": core_tags, "bonus": bonus_tags
    }

# (2) 기존 일봉 다이버전스 로직
def analyze_divergence(ticker: str, df: pd.DataFrame, name: str, market: str) -> Optional[dict]:
    if len(df) < 30: return None
    recent = df.tail(10)
    prev_period = df.iloc[-25:-10]
    
    curr_low_price = recent["Low"].min()
    prev_low_price = prev_period["Low"].min()
    
    curr_rsi = recent.loc[recent["Low"].idxmin(), "RSI"]
    prev_rsi = prev_period.loc[prev_period["Low"].idxmin(), "RSI"]
    
    if curr_low_price < prev_low_price and curr_rsi > prev_rsi and curr_rsi < 40:
        curr = df.iloc[-1]
        return {
            "name": name, "price": curr["Close"], "rsi": curr_rsi,
            "change": ((curr["Close"]/df.iloc[-2]["Close"])-1)*100
        }
    return None

# (3) 신규: 주봉 기준 볼린저 밴드 하단 재진입 로직
def analyze_bb_reentry_weekly(df_daily: pd.DataFrame, name: str) -> Optional[dict]:
    """
    일봉 데이터를 주봉으로 리샘플링하여 BB 하단 재진입 여부 판단
    """
    if len(df_daily) < 100: return None
    
    # 주봉 변환 (Open: 첫값, High: 최대, Low: 최소, Close: 마지막, Volume: 합계)
    df_weekly = df_daily.resample('W').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    
    if len(df_weekly) < 20: return None
    
    # 주봉 지표 계산
    df_weekly["MA20"] = df_weekly["Close"].rolling(20).mean()
    df_weekly["STD"] = df_weekly["Close"].rolling(20).std()
    df_weekly["BB_LOWER"] = df_weekly["MA20"] - (df_weekly["STD"] * 2)
    
    w_curr = df_weekly.iloc[-1]
    w_prev = df_weekly.iloc[-2]
    w_prev2 = df_weekly.iloc[-3]
    
    reentry = False
    # 이번 주 재진입 (지난 주 하단 아래 -> 이번 주 현재 하단 위)
    if w_prev["Close"] < w_prev["BB_LOWER"] and w_curr["Close"] > w_curr["BB_LOWER"]:
        reentry = True
    # 지난 주 재진입 (지지난 주 하단 아래 -> 지난 주 하단 위 마감)
    elif w_prev2["Close"] < w_prev2["BB_LOWER"] and w_prev["Close"] > w_prev["BB_LOWER"] and w_curr["Close"] > w_curr["BB_LOWER"]:
        reentry = True
        
    if reentry:
        return {
            "name": name, "price": w_curr["Close"],
            "change": ((w_curr["Close"]/w_prev["Close"])-1)*100,
            "lower": w_curr["BB_LOWER"]
        }
    return None

# =========================================================
# 4. 전송 및 실행
# =========================================================

def send_telegram(msg: str):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try: requests.post(url, json=payload, timeout=20)
    except: pass

def process_market(market_name: str, tickers: list, names: dict, is_bull: bool):
    logger.info(f"[{market_name}] 데이터 분석 시작...")
    try:
        # 주봉 분석을 위해 충분한 기간(2년) 데이터 수집
        data = yf.download(tickers, period="2y", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    except: return

    score_results = []
    div_results = []
    bb_weekly_results = []

    for ticker in tickers:
        try:
            if ticker not in data.columns.levels[0]: continue
            df = data[ticker].dropna()
            if df.empty: continue
            
            # 일봉 지표 계산 (기존 로직용)
            df_daily = calculate_indicators(df)

            # 1. 일봉 스코어링
            s_res = analyze_logic(ticker, df_daily, names.get(ticker, ticker), market_name, is_bull)
            if s_res: score_results.append(s_res)

            # 2. 일봉 다이버전스
            d_res = analyze_divergence(ticker, df_daily, names.get(ticker, ticker), market_name)
            if d_res: div_results.append(d_res)
            
            # 3. 주봉 BB 하단 재진입 (신규)
            bb_res = analyze_bb_reentry_weekly(df, names.get(ticker, ticker))
            if bb_res: bb_weekly_results.append(bb_res)
            
        except: continue

    flag = "🇰🇷" if market_name == "KOREA" else "🇺🇸"
    unit = "₩" if market_name == "KOREA" else "$"

    # 메시지 1: 일봉 스코어 분석
    score_results = sorted(score_results, key=lambda x: -x["score"])[:10]
    if score_results:
        msg = f"<b>{flag} {market_name} 일봉 바닥 반등</b>\n"
        msg += f"<i>추세: {'🔵상승/횡보' if is_bull else '🔴하락(보정)'}</i>\n\n"
        for i, r in enumerate(score_results):
            core_str = " ".join([f"#{t}" for t in r["core"]])
            bonus_str = f" <code>[{', '.join(r['bonus'])}]</code>" if r["bonus"] else ""
            msg += f"{i+1}. <b>{html.escape(r['name'])}</b> ({r['score']}점){bonus_str}\n"
            msg += f"└ 💰 {unit}{r['price']:,.0f} ({r['change']:+.1f}%) | RSI:{r['rsi']:.0f}\n"
            msg += f"└ 📊 {core_str}\n\n"
        send_telegram(msg)

    # 메시지 2: 일봉 다이버전스 분석
    div_results = div_results[:10]
    if div_results:
        msg = f"<b>🔍 {market_name} 일봉 상승 다이버전스</b>\n"
        msg += f"<i>대상: 40 미만 저점 반등 포착</i>\n\n"
        for i, r in enumerate(div_results):
            msg += f"{i+1}. <b>{html.escape(r['name'])}</b>\n"
            msg += f"└ 💰 {unit}{r['price']:,.0f} ({r['change']:+.1f}%) | RSI:{r['rsi']:.1f}\n\n"
        send_telegram(msg)
        
    # 메시지 3: 주봉 볼린저 밴드 하단 재진입
    if bb_weekly_results:
        bb_weekly_results = bb_weekly_results[:10]
        msg = f"<b>🛡️ {market_name} 주봉 BB 하단 재진입</b>\n"
        msg += f"<i>대상: 주봉 하단 이탈 후 안쪽 복귀(이번주/지난주)</i>\n\n"
        for i, r in enumerate(bb_weekly_results):
            msg += f"{i+1}. <b>{html.escape(r['name'])}</b>\n"
            msg += f"└ 💰 {unit}{r['price']:,.0f} ({r['change']:+.1f}%) | 주봉하단:{r['lower']:,.0f}\n\n"
        send_telegram(msg)

def main():
    regime = get_market_regime()
    # KOREA
    try:
        kor = fdr.StockListing("KRX").sort_values("Marcap", ascending=False).head(KOR_TOP_N)
        kor_tickers = [str(c) + (".KS" if m == "KOSPI" else ".KQ") for c, m in zip(kor["Code"], kor["Market"])]
        process_market("KOREA", kor_tickers, dict(zip(kor_tickers, kor["Name"])), regime["KOREA"])
    except: pass

    # USA
    try:
        us = fdr.StockListing("S&P500").head(USA_TOP_N)
        us_names = {str(row["Symbol"]).replace(".", "-"): row["Name"] for _, row in us.iterrows()}
        process_market("USA", list(us_names.keys()), us_names, regime["USA"])
    except: pass

if __name__ == "__main__":
    main()
