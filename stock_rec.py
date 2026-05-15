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

# (3) 기존 일봉 기준 볼린저 밴드 하단 재진입 로직
def analyze_bb_reentry_daily(df: pd.DataFrame, name: str) -> Optional[dict]:
    if len(df) < 25: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    
    is_reentry = prev["Close"] < prev["BB_LOWER"] and curr["Close"] > curr["BB_LOWER"]
    is_bullish = curr["Close"] > curr["Open"]
    is_vol_up = curr["Volume"] > prev["Volume"]
    
    if is_reentry and is_bullish and is_vol_up:
        return {
            "name": name, "price": curr["Close"],
            "change": ((curr["Close"]/prev["Close"])-1)*100,
            "lower": curr["BB_LOWER"],
            "vol_ratio": (curr["Volume"] / prev["Volume"]) * 100
        }
    return None

# (4) 5분봉 스케일로 전면 수정된 적삼병 후 눌림목 반등 로직
def analyze_three_soldiers_pullback_5m(df_5m: pd.DataFrame, name: str) -> Optional[dict]:
    """
    5분봉 기준: 3연속 양봉(강한 상승) 후, 2~3개 캔들 동안 거래량이 줄어들며 
    상승폭의 중심선 부근까지 눌렸다가 현재 5분봉에서 양봉 반등하는 종목 포착
    """
    if len(df_5m) < 15: return None
    
    # 현재 미완성 5분봉 캔들도 양봉(현재가 > 시가)이어야 실시간 진입 성립
    curr = df_5m.iloc[-1]
    if curr["Close"] <= curr["Open"]: return None
    
    def check_pattern(pullback_candles: int) -> Optional[dict]:
        total_candles = 3 + pullback_candles + 1
        sub_df = df_5m.iloc[-total_candles:]
        
        # 1. 5분봉 적삼병 구간 (연속 3개 양봉)
        three_bulls = sub_df.iloc[0:3]
        for _, row in three_bulls.iterrows():
            if row["Close"] <= row["Open"]: return None
            
        red_low = three_bulls["Low"].min()
        red_high = three_bulls["High"].max()
        red_mid = (red_low + red_high) / 2 # 5분봉 기준 피보나치 0.5 지지선
        red_vol_avg = three_bulls["Volume"].mean()
        
        # 2. 5분봉 음봉 조정 구간 (2~3개 캔들 연속 음봉)
        bears = sub_df.iloc[3:3+pullback_candles]
        for _, row in bears.iterrows():
            if row["Close"] >= row["Open"]: return None
            # 분봉 노이즈를 감안해 음봉 거래량이 상승 평균 거래량의 90% 미만인지 체크 (거래량 감소)
            if row["Volume"] >= red_vol_avg * 0.90: return None 
            if row["Close"] < red_mid * 0.995: return None # 중심선 이탈 방지
            
        # 3. 현재 5분봉 지지 확인
        if curr["Low"] < red_mid * 0.99: return None
        
        return {
            "name": name, "price": curr["Close"],
            "change": ((curr["Close"] / df_5m.iloc[-2]["Close"]) - 1) * 100,
            "pullback_candles": pullback_candles,
            "mid_price": red_mid
        }

    result = check_pattern(pullback_candles=2)
    if not result:
        result = check_pattern(pullback_candles=3)
        
    return result

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
    logger.info(f"[{market_name}] 일봉 및 5분봉 데이터 다운로드 중...")
    
    # 1. 기존 일봉 데이터 다운로드 (1, 2, 3번 로직용)
    try:
        data_daily = yf.download(tickers, period="12mo", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    except: data_daily = pd.DataFrame()

    # 2. 신규 5분봉 데이터 다운로드 (4번 로직용 / 최대 5일 분량 제한)
    try:
        data_5m = yf.download(tickers, period="5d", interval="5m", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    except: data_5m = pd.DataFrame()

    score_results = []
    div_results = []
    bb_daily_results = []
    pullback_results = []

    for ticker in tickers:
        name = names.get(ticker, ticker)
        
        # 일봉 기반 분석 실행
        if not data_daily.empty and ticker in data_daily.columns.levels[0]:
            try:
                df_d = data_daily[ticker].dropna()
                if not df_d.empty:
                    df_d = calculate_indicators(df_d)
                    
                    s_res = analyze_logic(ticker, df_d, name, market_name, is_bull)
                    if s_res: score_results.append(s_res)

                    d_res = analyze_divergence(ticker, df_d, name, market_name)
                    if d_res: div_results.append(d_res)
                    
                    bb_res = analyze_bb_reentry_daily(df_d, name)
                    if bb_res: bb_daily_results.append(bb_res)
            except: pass

        # 5분봉 기반 분석 실행 (4번 로직)
        if not data_5m.empty and ticker in data_5m.columns.levels[0]:
            try:
                df_5 = data_5m[ticker].dropna()
                if not df_5.empty:
                    pb_res = analyze_three_soldiers_pullback_5m(df_5, name)
                    if pb_res: pullback_results.append(pb_res)
            except: pass

    flag = "🇰🇷" if market_name == "KOREA" else "🇺🇸"
    unit = "₩" if market_name == "KOREA" else "$"

    # 메시지 1: 일봉 스코어 분석
    score_results = sorted(score_results, key=lambda x: -x["score"])[:10]
    if score_results:
        msg = f"<b>{flag} {market_name} 일봉 바닥 반등</b>\n"
        for i, r in enumerate(score_results):
            core_str = " ".join([f"#{t}" for t in r["core"]])
            msg += f"{i+1}. <b>{html.escape(r['name'])}</b> ({r['score']}점)\n"
            msg += f"└ 💰 {unit}{r['price']:,.0f} ({r['change']:+.1f}%) | RSI:{r['rsi']:.0f}\n"
            msg += f"└ 📊 {core_str}\n\n"
        send_telegram(msg)

    # 메시지 2: 일봉 다이버전스 분석
    div_results = div_results[:10]
    if div_results:
        msg = f"<b>🔍 {market_name} 일봉 상승 다이버전스</b>\n"
        for i, r in enumerate(div_results):
            msg += f"{i+1}. <b>{html.escape(r['name'])}</b>\n"
            msg += f"└ 💰 {unit}{r['price']:,.0f} ({r['change']:+.1f}%) | RSI:{r['rsi']:.1f}\n\n"
        send_telegram(msg)
        
    # 메시지 3: 일봉 BB 하단 재진입
    if bb_daily_results:
        bb_daily_results = bb_daily_results[:10]
        msg = f"<b>🛡️ {market_name} 일봉 BB 재진입 (검증)</b>\n"
        msg += f"<i>조건: 양봉 마감 + 전일대비 거래량 증가</i>\n\n"
        for i, r in enumerate(bb_daily_results):
            msg += f"{i+1}. <b>{html.escape(r['name'])}</b>\n"
            msg += f"└ 💰 {unit}{r['price']:,.0f} ({r['change']:+.1f}%)\n"
            msg += f"└ 📊 거래량비: {r['vol_ratio']:.0f}% | 하단선:{r['lower']:,.0f}\n\n"
        send_telegram(msg)

    # 메시지 4: 5분봉 적삼병 후 거래량 감소 눌림목 반등 (수정됨)
    if pullback_results:
        pullback_results = pullback_results[:10]
        msg = f"<b>⚡ {market_name} 5분봉 눌림목 타점 포착</b>\n"
        msg += f"<i>조건: 5분봉 3연속 양봉 후 거래량 급감 조정 + 중심선 지지 반등</i>\n\n"
        for i, r in enumerate(pullback_results):
            msg += f"{i+1}. <b>{html.escape(r['name'])}</b>\n"
            msg += f"└ 💰 현재가: {unit}{r['price']:,.0f} ({r['change']:+.1f}%)\n"
            msg += f"└ 📊 5분봉 {r['pullback_candles']}개 음봉 조정 후 양봉 턴어라운드 | 기준선:{r['mid_price']:,.0f}\n\n"
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
