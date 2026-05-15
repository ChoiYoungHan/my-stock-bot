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
    """
    일봉 기준 BB 하단 재진입 판별 (거래량 증가 + 양봉 조건 추가)
    """
    if len(df) < 25: return None
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
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

# (4) 신규 추가: 적삼병 후 거래량 감소 눌림목 반등 로직
def analyze_three_soldiers_pullback(df: pd.DataFrame, name: str) -> Optional[dict]:
    """
    적삼병(3양봉) 후 2~3일 음봉 조정을 거치되, 
    조정 시 거래량이 급감하고 적삼병 중심선을 지키며 당일 양봉 전환(반등)하는 종목 포착
    """
    if len(df) < 10: return None
    
    # 최근 7거래일 데이터 확인 (음봉 2일 조정 케이스 vs 음봉 3일 조정 케이스 검사)
    # 오늘(Index -1)은 무조건 반등 양봉이어야 함
    curr = df.iloc[-1]
    if curr["Close"] <= curr["Open"]: return None
    
    # 패턴 매칭용 헬퍼 변수
    def check_pattern(pullback_days: int) -> Optional[dict]:
        # 전체 패턴 길이 = 적삼병(3일) + 조정(2~3일) + 당일반등(1일)
        total_days = 3 + pullback_days + 1
        sub_df = df.iloc[-total_days:]
        
        # 1. 적삼병 구간 (구간 내 인덱스 0, 1, 2)
        three_bulls = sub_df.iloc[0:3]
        for _, row in three_bulls.iterrows():
            if row["Close"] <= row["Open"]: return None # 3일 연속 양봉이어야 함
            
        # 적삼병 구간의 전체 고가, 저가, 평균 거래량 계산
        red_low = three_bulls["Low"].min()
        red_high = three_bulls["High"].max()
        red_mid = (red_low + red_high) / 2 # 적삼병의 중간값 (지지선)
        red_vol_avg = three_bulls["Volume"].mean()
        
        # 2. 음봉 조정 구간 (구간 내 인덱스 3부터 3+pullback_days까지)
        bears = sub_df.iloc[3:3+pullback_days]
        for _, row in bears.iterrows():
            if row["Close"] >= row["Open"]: return None # 조정일은 모두 음봉이어야 함
            # [필터] 조정 시 음봉 거래량이 적삼병 평균 거래량보다 작아야 함 (거래량 감소 눌림목)
            if row["Volume"] >= red_vol_avg * 0.85: return None 
            # [필터] 조정 시 종가가 적삼병 중심선 아래로 과도하게 밀리면 제외
            if row["Close"] < red_mid: return None
            
        # 3. 오늘(최종 반등일)의 검증
        # 저가가 중심선 부근에서 지지받고 올라왔는지 확인
        if curr["Low"] < red_mid * 0.98: return None # 중심선을 너무 깊게 깨면 제외
        
        return {
            "name": name, "price": curr["Close"],
            "change": ((curr["Close"] / df.iloc[-2]["Close"]) - 1) * 100,
            "pullback_days": pullback_days,
            "mid_price": red_mid
        }

    # 음봉 2일 조정 후 반등 패턴 먼저 체크 -> 없으면 3일 조정 체크
    result = check_pattern(pullback_days=2)
    if not result:
        result = check_pattern(pullback_days=3)
        
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
    logger.info(f"[{market_name}] 데이터 분석 시작...")
    try:
        data = yf.download(tickers, period="12mo", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    except: return

    score_results = []
    div_results = []
    bb_daily_results = []
    pullback_results = [] # 신규 결과 배열

    for ticker in tickers:
        try:
            if ticker not in data.columns.levels[0]: continue
            df = data[ticker].dropna()
            if df.empty: continue
            
            # 공통 지표 계산
            df = calculate_indicators(df)

            # 1. 일봉 스코어링
            s_res = analyze_logic(ticker, df, names.get(ticker, ticker), market_name, is_bull)
            if s_res: score_results.append(s_res)

            # 2. 일봉 다이버전스
            d_res = analyze_divergence(ticker, df, names.get(ticker, ticker), market_name)
            if d_res: div_results.append(d_res)
            
            # 3. 일봉 BB 하단 재진입 (보완 로직)
            bb_res = analyze_bb_reentry_daily(df, names.get(ticker, ticker))
            if bb_res: bb_daily_results.append(bb_res)
            
            # 4. 일봉 적삼병 후 눌림목 반등 (신규 로직)
            pb_res = analyze_three_soldiers_pullback(df, names.get(ticker, ticker))
            if pb_res: pullback_results.append(pb_res)
            
        except: continue

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
        
    # 메시지 3: 일봉 BB 하단 재진입 (보완 로직)
    if bb_daily_results:
        bb_daily_results = bb_daily_results[:10]
        msg = f"<b>🛡️ {market_name} 일봉 BB 재진입 (검증)</b>\n"
        msg += f"<i>조건: 양봉 마감 + 전일대비 거래량 증가</i>\n\n"
        for i, r in enumerate(bb_daily_results):
            msg += f"{i+1}. <b>{html.escape(r['name'])}</b>\n"
            msg += f"└ 💰 {unit}{r['price']:,.0f} ({r['change']:+.1f}%)\n"
            msg += f"└ 📊 거래량비: {r['vol_ratio']:.0f}% | 하단선:{r['lower']:,.0f}\n\n"
        send_telegram(msg)

    # 메시지 4: 일봉 적삼병 후 거래량 감소 눌림목 반등 (신규 메세지)
    if pullback_results:
        pullback_results = pullback_results[:10]
        msg = f"<b>📈 {market_name} 적삼병 후 눌림목 반등</b>\n"
        msg += f"<i>조건: 3양봉 후 거래량 급감 2~3음봉 조정 + 중심선 지지 반등</i>\n\n"
        for i, r in enumerate(pullback_results):
            msg += f"{i+1}. <b>{html.escape(r['name'])}</b>\n"
            msg += f"└ 💰 {unit}{r['price']:,.0f} ({r['change']:+.1f}%)\n"
            msg += f"└ 📊 {r['pullback_days']}일간 음봉 조정 후 오늘 양봉 전환 | 기준선:{r['mid_price']:,.0f}\n\n"
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
