import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
import os

# =========================================================
# 설정
# =========================================================
logging.getLogger("yf").setLevel(logging.CRITICAL)

TOKEN   = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

KOR_TOP_N       = 500
USA_TOP_N       = 500
MIN_SCORE       = 55
MAX_DAILY_CHANGE = 10.0   # 당일 급등 제외
MAX_5D_CHANGE   = 25.0    # 5일 급등 제외


# =========================================================
# 텔레그램
# =========================================================
def send_telegram(msg: str):
    if not (TOKEN and CHAT_ID):
        print(msg)
        return
    url     = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg}
    try:
        requests.post(url, json=payload, timeout=20)
    except Exception:
        pass


# =========================================================
# 지표 계산
# =========================================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    # --------------------------------------------------
    # 이동평균
    # --------------------------------------------------
    df["MA5"]  = c.rolling(5).mean()
    df["MA20"] = c.rolling(20).mean()
    df["MA60"] = c.rolling(60).mean()

    # --------------------------------------------------
    # 볼린저 밴드 + Squeeze(수축)
    # --------------------------------------------------
    std20          = c.rolling(20).std()
    df["BB_UPPER"] = df["MA20"] + std20 * 2
    df["BB_LOWER"] = df["MA20"] - std20 * 2
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["MA20"]
    # BB Width의 60일 이동평균 → Squeeze 판단 기준
    df["BB_WIDTH_MA60"] = df["BB_WIDTH"].rolling(60).mean()

    # --------------------------------------------------
    # 거래량 관련
    # --------------------------------------------------
    df["VMA20"]    = v.rolling(20).mean()
    df["VOL_RATIO"] = v / df["VMA20"]

    # --------------------------------------------------
    # RSI (14)
    # --------------------------------------------------
    delta    = c.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # --------------------------------------------------
    # MACD (12/26/9)
    # --------------------------------------------------
    ema12           = c.ewm(span=12, adjust=False).mean()
    ema26           = c.ewm(span=26, adjust=False).mean()
    df["MACD"]      = ema12 - ema26
    df["MACD_SIG"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIG"]

    # --------------------------------------------------
    # 이격도
    # --------------------------------------------------
    df["DISPARITY20"] = (c / df["MA20"]) * 100

    # --------------------------------------------------
    # 스토캐스틱 (14, 3)  [신규]
    # --------------------------------------------------
    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df["STOCH_K"] = (c - low14) / (high14 - low14 + 1e-9) * 100
    df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()

    # --------------------------------------------------
    # Williams %R (14)  [신규]
    # --------------------------------------------------
    df["WR"] = (high14 - c) / (high14 - low14 + 1e-9) * -100

    # --------------------------------------------------
    # OBV  [신규]
    # --------------------------------------------------
    direction   = np.sign(c.diff()).fillna(0)
    df["OBV"]   = (v * direction).cumsum()
    df["OBV_MA5"] = df["OBV"].rolling(5).mean()

    # --------------------------------------------------
    # ATR (14)
    # --------------------------------------------------
    tr1       = h - l
    tr2       = (h - c.shift(1)).abs()
    tr3       = (l - c.shift(1)).abs()
    tr        = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    return df


# =========================================================
# 캔들 패턴 유틸
# =========================================================
def candle_strength(row) -> float:
    """몸통 비율 (0~1)"""
    body  = abs(row["Close"] - row["Open"])
    total = row["High"] - row["Low"]
    return 0.0 if total == 0 else body / total


def is_doji(row, threshold: float = 0.15) -> bool:
    """
    [신규] 도지 캔들:
    몸통이 고-저 범위의 15% 미만
    """
    body  = abs(row["Close"] - row["Open"])
    total = row["High"] - row["Low"]
    if total == 0:
        return False
    return (body / total) < threshold


def is_hammer(row) -> bool:
    """
    [신규] 해머(망치형):
    - 아래 꼬리 >= 몸통 * 2
    - 위 꼬리 <= 몸통 * 0.5
    - 양봉/음봉 무관 (바닥권 전제)
    """
    open_  = row["Open"]
    close  = row["Close"]
    high   = row["High"]
    low    = row["Low"]
    body   = abs(close - open_)
    if body == 0:
        body = 1e-9
    lower_wick = min(open_, close) - low
    upper_wick = high - max(open_, close)
    return (lower_wick >= body * 2) and (upper_wick <= body * 0.5)


def is_inverted_hammer(row, prev_row) -> bool:
    """
    [신규] 역망치(역해머):
    - 위 꼬리 >= 몸통 * 2
    - 아래 꼬리 <= 몸통 * 0.5
    - 전일 음봉 조건
    """
    open_  = row["Open"]
    close  = row["Close"]
    high   = row["High"]
    low    = row["Low"]
    body   = abs(close - open_)
    if body == 0:
        body = 1e-9
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    prev_bearish = prev_row["Close"] < prev_row["Open"]
    return (upper_wick >= body * 2) and (lower_wick <= body * 0.5) and prev_bearish


def count_consecutive_down(df: pd.DataFrame, lookback: int = 10) -> int:
    """
    [신규] 연속 음봉 일수 계산 (최근 기준 역순)
    """
    count = 0
    closes = df["Close"].values
    opens  = df["Open"].values
    # 최근 캔들 제외하고 그 이전부터 역순
    for i in range(len(df) - 2, max(len(df) - 2 - lookback, -1), -1):
        if closes[i] < opens[i]:
            count += 1
        else:
            break
    return count


def is_base_pattern(df: pd.DataFrame, window: int = 10, cv_thresh: float = 0.02) -> bool:
    """
    [신규] 바닥 다지기 패턴:
    최근 N일간 저점의 변동계수(CV)가 낮으면 횡보 수렴 (바닥 다지기)
    """
    recent_lows = df["Low"].iloc[-window:]
    mean_low    = recent_lows.mean()
    if mean_low == 0:
        return False
    cv = recent_lows.std() / mean_low
    return cv < cv_thresh


# =========================================================
# 분석 로직
# =========================================================
def analyze_logic(ticker: str, df: pd.DataFrame, name: str, market: str) -> dict | None:
    if len(df) < 120:
        return None

    df   = calculate_indicators(df)
    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # --------------------------------------------------
    # 당일 급등 제외
    # --------------------------------------------------
    change = ((curr["Close"] / prev["Close"]) - 1) * 100
    if change >= MAX_DAILY_CHANGE:
        return None

    # --------------------------------------------------
    # 유동성 필터 (한국/미국 기준 분리)  [개선]
    # --------------------------------------------------
    vol_money = curr["Volume"] * curr["Close"]
    if market == "KOREA":
        if vol_money < 5_000_000_000:   # 50억 원
            return None
    else:
        if vol_money < 5_000_000:       # 500만 달러
            return None

    # --------------------------------------------------
    # 5일 급등 제외
    # --------------------------------------------------
    if len(df) >= 6:
        recent_5d = ((curr["Close"] / df.iloc[-6]["Close"]) - 1) * 100
        if recent_5d >= MAX_5D_CHANGE:
            return None

    # ==================================================
    # 점수 계산
    # ==================================================
    score = 0
    tags  = []

    # --------------------------------------------------
    # 1. 볼린저 하단 근접 (BB_LOWER * 1.04 이내)  [기존]
    # --------------------------------------------------
    bb_near = (prev["Close"] <= prev["BB_LOWER"] * 1.04)
    if bb_near:
        score += 15
        tags.append("BB")

    # --------------------------------------------------
    # 2. 볼린저 하단 터치 후 종가 회복  [신규]
    #    장중 BB_LOWER 하회 → 종가는 MA20 위
    # --------------------------------------------------
    bb_reversal = (
        curr["Low"] < curr["BB_LOWER"]
        and curr["Close"] > curr["MA20"]
    )
    if bb_reversal:
        score += 20
        tags.append("BB_REV")

    # --------------------------------------------------
    # 3. 볼린저 밴드 Squeeze (수축 후 확장 직전)  [신규]
    #    BB Width < 60일 평균의 70%
    # --------------------------------------------------
    try:
        squeeze = (
            pd.notna(curr["BB_WIDTH_MA60"])
            and curr["BB_WIDTH"] < curr["BB_WIDTH_MA60"] * 0.70
        )
    except Exception:
        squeeze = False
    if squeeze:
        score += 10
        tags.append("SQUEEZE")

    # --------------------------------------------------
    # 4. RSI 반등 (14)  [기존 + 기준 강화]
    #    이전 RSI < 35, 현재 RSI 상승
    # --------------------------------------------------
    rsi_rebound = (prev["RSI"] < 35) and (curr["RSI"] > prev["RSI"])
    if rsi_rebound:
        score += 20
        tags.append("RSI")

    # --------------------------------------------------
    # 5. 스토캐스틱 골든크로스 (과매도 구간)  [신규]
    #    %K < 20 + %K > %D 상향 돌파
    # --------------------------------------------------
    stoch_signal = (
        prev["STOCH_K"] < 20
        and curr["STOCH_K"] > curr["STOCH_D"]
        and prev["STOCH_K"] <= prev["STOCH_D"]
    )
    if stoch_signal:
        score += 15
        tags.append("STOCH")

    # --------------------------------------------------
    # 6. Williams %R 반등  [신규]
    #    이전 WR < -80 (과매도), 현재 WR 상승
    # --------------------------------------------------
    wr_rebound = (prev["WR"] < -80) and (curr["WR"] > prev["WR"])
    if wr_rebound:
        score += 10
        tags.append("WR")

    # --------------------------------------------------
    # 7. MACD 골든크로스  [기존]
    # --------------------------------------------------
    macd_cross = (
        prev["MACD"] <= prev["MACD_SIG"]
        and curr["MACD"] > curr["MACD_SIG"]
    )
    if macd_cross:
        score += 20
        tags.append("MACD")

    # --------------------------------------------------
    # 8. MACD 히스토그램 연속 증가 (수렴 → 반전 전조)  [신규]
    # --------------------------------------------------
    macd_hist_rising = (
        df["MACD_HIST"].iloc[-3] < df["MACD_HIST"].iloc[-2] < df["MACD_HIST"].iloc[-1]
    )
    if macd_hist_rising:
        score += 8
        tags.append("MACD_H")

    # --------------------------------------------------
    # 9. 거래량 급증  [기존]
    # --------------------------------------------------
    volume_surge = (curr["VOL_RATIO"] >= 1.5)   # 기준 상향 (1.3 → 1.5)
    if volume_surge:
        score += 20
        tags.append("VOL")

    # --------------------------------------------------
    # 10. OBV 상승 반전  [신규]
    #     OBV가 5일 이동평균 위로 복귀
    # --------------------------------------------------
    obv_up = (
        prev["OBV"] <= prev["OBV_MA5"]
        and curr["OBV"] > curr["OBV_MA5"]
    )
    if obv_up:
        score += 12
        tags.append("OBV")

    # --------------------------------------------------
    # 11. 도지 캔들  [신규]
    # --------------------------------------------------
    if is_doji(curr):
        score += 10
        tags.append("DOJI")

    # --------------------------------------------------
    # 12. 해머 캔들  [신규]
    # --------------------------------------------------
    if is_hammer(curr):
        score += 15
        tags.append("HAMMER")

    # --------------------------------------------------
    # 13. 역망치 캔들  [신규]
    # --------------------------------------------------
    if is_inverted_hammer(curr, prev):
        score += 10
        tags.append("INV_HAMMER")

    # --------------------------------------------------
    # 14. 장대양봉  [기존]
    # --------------------------------------------------
    bullish_candle = (
        curr["Close"] > curr["Open"]
        and candle_strength(curr) > 0.6
    )
    if bullish_candle:
        score += 15
        tags.append("CANDLE")
    elif curr["Close"] > curr["Open"]:
        # 일반 양봉
        score += 5

    # --------------------------------------------------
    # 15. MA5/MA20 골든크로스  [기존]
    # --------------------------------------------------
    ma_gc = (
        curr["MA5"] > curr["MA20"]
        and prev["MA5"] <= prev["MA20"]
    )
    if ma_gc:
        score += 20
        tags.append("GC")

    # --------------------------------------------------
    # 16. 이격도 과매도  [기존, 기준 소폭 조정]
    # --------------------------------------------------
    oversold = (curr["DISPARITY20"] < 93)
    if oversold:
        score += 10
        tags.append("OS")

    # --------------------------------------------------
    # 17. 52주 저점 근접  [신규]
    #     현재가가 52주 저가 대비 10% 이내
    # --------------------------------------------------
    low_52w = df["Low"].rolling(252).min().iloc[-1]
    near_52w_low = (curr["Close"] <= low_52w * 1.10)
    if near_52w_low:
        score += 12
        tags.append("52WL")

    # --------------------------------------------------
    # 18. 바닥 다지기 패턴  [신규]
    # --------------------------------------------------
    if is_base_pattern(df, window=10, cv_thresh=0.02):
        score += 12
        tags.append("BASE")

    # --------------------------------------------------
    # 19. 연속 하락 후 첫 양봉  [신규]
    # --------------------------------------------------
    consec_down = count_consecutive_down(df, lookback=8)
    if consec_down >= 3 and curr["Close"] > curr["Open"]:
        score += 15
        tags.append("REBOUND")

    # ==================================================
    # 최소 점수 기준
    # ==================================================
    if score < MIN_SCORE:
        return None

    return {
        "name"   : name,
        "score"  : score,
        "price"  : curr["Close"],
        "change" : change,
        "rsi"    : curr["RSI"],
        "stoch_k": curr["STOCH_K"],
        "wr"     : curr["WR"],
        "vol"    : curr["VOL_RATIO"],
        "tags"   : ",".join(tags),
    }


# =========================================================
# 시장 분석
# =========================================================
def process_market(market_name: str, tickers: list, names: dict):
    print(f"[{market_name}] 분석 시작 ({len(tickers)}종목)")

    try:
        data = yf.download(
            tickers,
            period="14mo",          # 52주 저점 계산 위해 기간 연장
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
    except Exception as e:
        print(f"[{market_name}] 데이터 다운로드 실패: {e}")
        return

    results = []
    for ticker in tickers:
        try:
            df = data[ticker].dropna()
            result = analyze_logic(ticker, df, names.get(ticker, ticker), market_name)
            if result:
                results.append(result)
        except Exception:
            continue

    # --------------------------------------------------
    # 정렬: 점수 내림차순 → RSI 오름차순
    # --------------------------------------------------
    results = sorted(results, key=lambda x: (-x["score"], x["rsi"]))

    # --------------------------------------------------
    # 텔레그램 메시지
    # --------------------------------------------------
    flag = "[KR]" if market_name == "KOREA" else "[US]"
    cur  = "W" if market_name == "KOREA" else "$"
    now  = (datetime.utcnow() + timedelta(hours=9)).strftime("%m/%d %H:%M")

    if not results:
        send_telegram(f"{flag} {market_name}\n조건 부합 종목 없음")
        return

    msg = f"{flag} {market_name} 바닥반등 후보 ({now})\n\n"
    for r in results[:15]:
        msg += (
            f"{r['name']}  [{r['score']}]\n"
            f"{cur}{r['price']:,.0f}  {r['change']:+.1f}%\n"
            f"RSI:{r['rsi']:.0f}  "
            f"K:{r['stoch_k']:.0f}  "
            f"WR:{r['wr']:.0f}  "
            f"V:{r['vol']:.1f}x\n"
            f"{r['tags']}\n\n"
        )

    send_telegram(msg)
    print(f"[{market_name}] 결과 {len(results)}종목 발송 완료")


# =========================================================
# 메인
# =========================================================
def main():
    # --------------------------------------------------
    # 한국 시장
    # --------------------------------------------------
    kor = (
        fdr.StockListing("KRX")
        .sort_values("Marcap", ascending=False)
        .head(KOR_TOP_N)
    )
    kor_tickers = []
    for _, row in kor.iterrows():
        suffix = ".KS" if row["Market"] == "KOSPI" else ".KQ"
        kor_tickers.append(row["Code"] + suffix)

    kor_names = dict(zip(kor_tickers, kor["Name"]))
    process_market("KOREA", kor_tickers, kor_names)

    # --------------------------------------------------
    # 미국 시장 (S&P500)
    # --------------------------------------------------
    us = fdr.StockListing("S&P500").head(USA_TOP_N)
    us_tickers = [t.replace(".", "-") for t in us["Symbol"]]
    us_names   = dict(zip(us_tickers, us["Name"]))
    process_market("USA", us_tickers, us_names)


# =========================================================
# 실행
# =========================================================
if __name__ == "__main__":
    main()
