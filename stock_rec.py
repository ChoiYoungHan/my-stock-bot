import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import html
import logging
from typing import Optional, Dict, List

# =========================================================
# 1. 로깅 설정 (에러 추적 및 로그 남기기)
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경 변수 (본인의 토큰과 ID로 설정 필요)
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

KOR_TOP_N = 500
USA_TOP_N = 500
MIN_SCORE = 40  # 상승장 기준 최소 점수

# =========================================================
# 6 & 7. 시장 추세 필터 (시장 방향성 확인)
# =========================================================
def get_market_regime() -> Dict[str, bool]:
    """지수 이평선을 이용해 현재 시장이 매매 적기(상승/횡보)인지 판단"""
    regime = {"KOREA": True, "USA": True}
    try:
        # 한국: KOSPI, 미국: S&P500(SPY)
        indices = {"KOREA": "KS11", "USA": "SPY"}
        for mkt, ticker in indices.items():
            df = fdr.DataReader(ticker, (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'))
            if df.empty: continue
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            curr_price = df['Close'].iloc[-1]
            # 지수가 20일선 위에 있으면 상승장(True), 아래면 하락장(False)
            regime[mkt] = curr_price > ma20
    except Exception as e:
        logger.error(f"시장 추세 분석 실패: {e}")
    return regime

# =========================================================
# 3. 기술적 지표 계산 (MACD adjust=False 적용)
# =========================================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    
    # 볼린저 밴드
    df["MA20"] = c.rolling(20).mean()
    df["BB_LOWER"] = df["MA20"] - (c.rolling(20).std() * 2)
    
    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    # MACD (표준 트레이딩 방식: adjust=False)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIG"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    # 거래대금 및 이평선
    df["Money"] = c * v
    df["MA5"] = c.rolling(5).mean()
    
    return df

# =========================================================
# 4 & 8. 분석 로직 (핵심 필터 + 부가 시그널 표기)
# =========================================================
def analyze_logic(ticker: str, df: pd.DataFrame, name: str, market: str, is_bull: bool) -> Optional[dict]:
    if len(df) < 50: return None
    df = calculate_indicators(df)
    curr, prev = df.iloc[-1], df.iloc[-2]
    
    # 2. 거래대금 필터 (잡주 제거: 한국 100억, 미국 1000만 달러 이상)
    min_money = 10_000_000_000 if market == "KOREA" else 10_000_000
    if curr["Money"] < min_money: return None

    # 8. 동적 기준 조정: 하락장(is_bull=False)일 때는 기준 점수를 높임
    dynamic_min_score = MIN_SCORE if is_bull else MIN_SCORE + 20
    
    score = 0
    core_tags = []    # 핵심 (점수 반영)
    bonus_tags = []   # 부가 (참고용)

    # --- 핵심 시그널 1: 볼린저 하단 지지 ---
    if curr["Low"] < curr["BB_LOWER"] * 1.01:
        score += 20
        core_tags.append("BB_SUPP")
        
    # --- 핵심 시그널 2: RSI 과매도 반등 ---
    if prev["RSI"] < 35 and curr["RSI"] > prev["RSI"]:
        score += 20
        core_tags.append("RSI_REV")
        
    # --- 핵심 시그널 3: MACD 골든크로스 ---
    if prev["MACD"] <= prev["MACD_SIG"] and curr["MACD"] > curr["MACD_SIG"]:
        score += 20
        core_tags.append("MACD_GC")

    # 필터링: 핵심 점수가 동적 기준 미달이면 제외
    if score < dynamic_min_score: return None

    # --- 부가 시그널 (도지, 이평선 등) ---
    # 1. 도지 캔들
    body = abs(curr["Close"] - curr["Open"])
    total_range = curr["High"] - curr["Low"]
    if total_range > 0 and (body / total_range) < 0.15:
        bonus_tags.append("DOJI")
        
    # 2. 단기 이평선 정배열 초기 (5일선 > 20일선)
    if curr["MA5"] > curr["MA20"]:
        bonus_tags.append("MA_UP")

    # 3. 강한 매수세 (윗꼬리가 짧은 양봉)
    if curr["Close"] > curr["Open"] and (curr["High"] - curr["Close"]) < body * 0.1:
        bonus_tags.append("STRONG")

    return {
        "name": name, "score": score, "price": curr["Close"],
        "change": ((curr["Close"]/prev["Close"])-1)*100,
        "rsi": curr["RSI"], "core": core_tags, "bonus": bonus_tags
    }

# =========================================================
# 텔레그램 전송 및 시장별 프로세스
# =========================================================
def send_telegram(msg: str):
    if not TOKEN or not CHAT_ID:
        logger.warning("텔레그램 설정이 없습니다.")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try:
        resp = requests.post(url, json=payload, timeout=20)
        if resp.status_code != 200: logger.error(f"전송 실패: {resp.text}")
    except Exception as e: logger.error(f"텔레그램 전송 중 오류: {e}")

def process_market(market_name: str, tickers: list, names: dict, is_bull: bool):
    logger.info(f"[{market_name}] 분석 시작... (추세: {'상승' if is_bull else '하락'})")
    
    try:
        # 데이터 일괄 다운로드
        data = yf.download(tickers, period="12mo", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    except Exception as e:
        logger.error(f"[{market_name}] 데이터 다운로드 치명적 오류: {e}")
        return

    results = []
    for ticker in tickers:
        try:
            # 1. except: continue 대신 개별 종목 예외 확인
            if ticker not in data.columns.levels[0]: continue
            
            df = data[ticker].dropna()
            if df.empty: continue
            
            res = analyze_logic(ticker, df, names.get(ticker, ticker), market_name, is_bull)
            if res: results.append(res)
        except Exception as e:
            logger.error(f"[{market_name}] 종목 분석 에러 ({ticker}): {e}")
            continue

    # 상위 10개 결과 전송
    results = sorted(results, key=lambda x: -x["score"])[:10]
    if not results: return

    now = (datetime.utcnow() + timedelta(hours=9)).strftime("%y/%m/%d %H:%M")
    flag = "🇰🇷" if market_name == "KOREA" else "🇺🇸"
    msg = f"<b>{flag} {market_name} 바닥 반등 TOP 10</b> ({now})\n"
    msg += f"<i>시장 추세: {'🔵상승/횡보' if is_bull else '🔴하락(기준강화)'}</i>\n\n"
    
    for i, r in enumerate(results):
        safe_name = html.escape(r["name"])
        core_str = " ".join([f"#{t}" for t in r["core"]])
        bonus_str = f" <code>[{', '.join(r['bonus'])}]</code>" if r["bonus"] else ""
        unit = "₩" if market_name == "KOREA" else "$"
        
        msg += f"{i+1}. <b>{safe_name}</b> ({r['score']}점){bonus_str}\n"
        msg += f"└ 💰 {unit}{r['price']:,.0f} ({r['change']:+.1f}%) | RSI:{r['rsi']:.0f}\n"
        msg += f"└ 📊 {core_str}\n\n"
    
    send_telegram(msg)

def main():
    regime = get_market_regime()

    # 1. 한국 시장
    try:
        kor = fdr.StockListing("KRX").sort_values("Marcap", ascending=False).head(KOR_TOP_N)
        kor_tickers = [str(c) + (".KS" if m == "KOSPI" else ".KQ") for c, m in zip(kor["Code"], kor["Market"])]
        process_market("KOREA", kor_tickers, dict(zip(kor_tickers, kor["Name"])), regime["KOREA"])
    except Exception as e:
        logger.error(f"한국 시장 프로세스 중단: {e}")

    # 2. 미국 시장
    try:
        us = fdr.StockListing("S&P500").head(USA_TOP_N)
        us_names = {str(row["Symbol"]).replace(".", "-"): row["Name"] for _, row in us.iterrows()}
        process_market("USA", list(us_names.keys()), us_names, regime["USA"])
    except Exception as e:
        logger.error(f"미국 시장 프로세스 중단: {e}")

if __name__ == "__main__":
    main()
