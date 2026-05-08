import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
import os
from typing import Optional

# =========================================================
# 설정 및 환경변수
# =========================================================
logging.getLogger("yf").setLevel(logging.CRITICAL)

# GitHub Secrets에 등록된 변수 읽기
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

KOR_TOP_N = 500
USA_TOP_N = 500
MIN_SCORE = 55
MAX_DAILY_CHANGE = 10.0
MAX_5D_CHANGE = 25.0

# =========================================================
# 텔레그램 전송 함수 (에러 로그 강화)
# =========================================================
def send_telegram(msg: str):
    if not (TOKEN and CHAT_ID):
        print("⚠️ 텔레그램 설정(TOKEN/ID)이 누락되었습니다. 콘솔에 출력합니다.")
        print(msg)
        return
    
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    
    try:
        resp = requests.post(url, json=payload, timeout=20)
        if resp.status_code == 200:
            print("✅ 텔레그램 메시지 전송 성공")
        else:
            print(f"❌ 전송 실패: {resp.text}")
    except Exception as e:
        print(f"❌ 텔레그램 전송 중 오류 발생: {e}")

# =========================================================
# 지표 계산 로직 (기존 로직 유지)
# =========================================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # 이동평균 및 볼린저 밴드
    df["MA5"] = c.rolling(5).mean()
    df["MA20"] = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["BB_UPPER"] = df["MA20"] + std20 * 2
    df["BB_LOWER"] = df["MA20"] - std20 * 2
    
    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIG"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Stochastic (K/D) & Williams %R
    low14, high14 = l.rolling(14).min(), h.rolling(14).max()
    df["STOCH_K"] = (c - low14) / (high14 - low14 + 1e-9) * 100
    df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()
    df["WR"] = (high14 - c) / (high14 - low14 + 1e-9) * -100

    # 거래량 비율
    df["VMA20"] = v.rolling(20).mean()
    df["VOL_RATIO"] = v / (df["VMA20"] + 1e-9)

    return df

# =========================================================
# 분석 메인 로직 (기존 점수제 유지)
# =========================================================
def analyze_logic(ticker: str, df: pd.DataFrame, name: str, market: str) -> Optional[dict]:
    if len(df) < 100: return None
    
    df = calculate_indicators(df)
    curr, prev = df.iloc[-1], df.iloc[-2]
    
    # 급등주 필터링
    change = ((curr["Close"] / prev["Close"]) - 1) * 100
    if change >= MAX_DAILY_CHANGE: return None
    
    # 유동성 필터링 (한국 50억, 미국 500만불)
    vol_money = curr["Volume"] * curr["Close"]
    money_limit = 5_000_000_000 if market == "KOREA" else 5_000_000
    if vol_money < money_limit: return None

    score = 0
    tags = []

    # 1. 볼린저 밴드 하단 터치 (바닥)
    if curr["Low"] < curr["BB_LOWER"] * 1.02:
        score += 15
        if curr["Close"] > curr["Open"]: # 하단 터치 후 양봉
            score += 10; tags.append("BB_UP")

    # 2. RSI / Stochastic / WR 반등
    if prev["RSI"] < 35 and curr["RSI"] > prev["RSI"]:
        score += 15; tags.append("RSI")
    if prev["STOCH_K"] < 20 and curr["STOCH_K"] > curr["STOCH_D"]:
        score += 15; tags.append("STOCH")
    if prev["WR"] < -80 and curr["WR"] > prev["WR"]:
        score += 10; tags.append("WR")

    # 3. MACD / 거래량 / 골든크로스
    if curr["MACD"] > curr["MACD_SIG"] and prev["MACD"] <= prev["MACD_SIG"]:
        score += 15; tags.append("MACD_GC")
    if curr["VOL_RATIO"] >= 1.5:
        score += 15; tags.append("VOL_UP")
    if curr["MA5"] > curr["MA20"] and prev["MA5"] <= prev["MA20"]:
        score += 15; tags.append("MA_GC")
    
    # 4. 도지 캔들 (하락 멈춤 신호)
    body = abs(curr["Close"] - curr["Open"])
    total = curr["High"] - curr["Low"]
    if total > 0 and (body / total) < 0.15 and (total > body * 2):
        score += 15; tags.append("DOJI")

    if score < MIN_SCORE: return None

    return {
        "name": name, "score": score, "price": curr["Close"],
        "change": change, "rsi": curr["RSI"], "vol": curr["VOL_RATIO"], "tags": tags[:3]
    }

# =========================================================
# 시장별 통합 프로세스
# =========================================================
def process_market(market_name: str, tickers: list, names: dict):
    print(f"[{market_name}] 분석 시작... 종목 수: {len(tickers)}")
    if not tickers: return

    try:
        data = yf.download(tickers, period="12mo", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    except Exception as e:
        print(f"데이터 다운로드 에러: {e}")
        return

    results = []
    for ticker in tickers:
        try:
            # 다중 종목 다운로드 시 데이터 구조 대응
            if len(tickers) > 1:
                df = data[ticker].dropna()
            else:
                df = data.dropna()
                
            if df.empty or len(df) < 50: continue
            
            res = analyze_logic(ticker, df, names.get(ticker, ticker), market_name)
            if res: results.append(res)
        except:
            continue

    # 상위 10개 정렬
    results = sorted(results, key=lambda x: -x["score"])[:10]
    
    # 텔레그램 메시지 생성
    now = (datetime.utcnow() + timedelta(hours=9)).strftime("%y/%m/%d %H:%M")
    flag = "🇰🇷" if market_name == "KOREA" else "🇺🇸"
    
    if not results:
        send_telegram(f"{flag} *{market_name}* ({now})\n조건 부합 종목 없음")
        return

    msg = f"{flag} *{market_name} 바닥 반등 TOP 10* ({now})\n\n"
    for i, r in enumerate(results):
        tag_str = " ".join([f"#{t}" for t in r["tags"]])
        cur_unit = "₩" if market_name == "KOREA" else "$"
        msg += f"{i+1}. *{r['name']}*  ({r['score']}점)\n"
        msg += f"└ 💰 {cur_unit}{r['price']:,.0f} ({r['change']:+.1f}%) | RSI:{r['rsi']:.0f}\n"
        msg += f"└ 📊 {tag_str}\n\n"

    send_telegram(msg)

# =========================================================
# 메인 실행부
# =========================================================
def main():
    # 1. 한국 시장
    try:
        print("KOREA 데이터 수집 중...")
        kor = fdr.StockListing("KRX").sort_values("Marcap", ascending=False).head(KOR_TOP_N)
        kor_tickers = [c + (".KS" if m == "KOSPI" else ".KQ") for c, m in zip(kor["Code"], kor["Market"])]
        kor_names = dict(zip(kor_tickers, kor["Name"]))
        process_market("KOREA", kor_tickers, kor_names)
    except Exception as e:
        print(f"한국 시장 분석 오류: {e}")

    # 2. 미국 시장 (S&P500)
    try:
        print("USA 데이터 수집 중...")
        us = fdr.StockListing("S&P500").head(USA_TOP_N)
        us_tickers = [t.replace(".", "-") for t in us["Symbol"]]
        us_names = dict(zip(us_tickers, us["Name"]))
        process_market("USA", us_tickers, us_names)
    except Exception as e:
        print(f"미국 시장 분석 오류: {e}")

if __name__ == "__main__":
    main()
