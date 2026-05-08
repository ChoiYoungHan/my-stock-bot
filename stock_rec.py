import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import logging
import os

# yfinance 불필요한 로그 출력 억제
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# --- [1. 설정 및 파라미터] ---
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(msg):
    """텔레그램 메시지 전송 함수埋"""
    if not (TOKEN and CHAT_ID):
        print("에러: TELEGRAM_TOKEN 또는 TELEGRAM_CHAT_ID 환경변수가 설정되지 않았습니다.")
        return
    
    if not msg: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code != 200:
            print(f"전송 실패: {response.text}")
    except Exception as e:
        print(f"네트워크 에러: {e}")

def calculate_indicators(df):
    """보조지표 계산 로직"""
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    df['MA5'] = c.rolling(5).mean()
    df['MA20'] = c.rolling(20).mean()
    df['MA60'] = c.rolling(60).mean()
    df['V_MA20'] = v.rolling(20).mean()
    
    # 불린저 밴드 계산 (20일, 2표준편차)
    std = c.rolling(20).std()
    df['BB_U'], df['BB_L'] = df['MA20'] + (std * 2), df['MA20'] - (std * 2)
    
    delta = c.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0], down[down > 0] = 0, 0
    df['RSI'] = 100 - (100 / (1 + (up.ewm(13).mean() / down.abs().ewm(13).mean())))
    
    df['Tenkan'] = (h.rolling(9).max() + l.rolling(9).min()) / 2
    df['Kijun'] = (h.rolling(26).max() + l.rolling(26).min()) / 2
    return df

def analyze_candle(row):
    """현재 봉(Candle)의 모양 분석"""
    o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
    total = h - l
    if total == 0: return ""
    
    body = abs(c - o)
    upper_tail = h - max(o, c)
    lower_tail = min(o, c) - l
    
    if lower_tail > total * 0.5: return "⚓아래꼬리(지지)"
    if upper_tail > total * 0.5: return "⚠️위꼬리(저항)"
    if body < total * 0.1: return "⚖️도지(변곡)"
    if c > o and body > total * 0.8: return "🔥장대양봉"
    
    return ""

def analyze_logic(ticker, df, name):
    """전략 판별 및 캔들 분석 통합"""
    if len(df) < 60: return None, None
    df = calculate_indicators(df)
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    d2 = df.iloc[-3]
    d3 = df.iloc[-4]
    
    candle_type = analyze_candle(curr)

    res = {
        "name": name, 
        "ticker": ticker, 
        "price": curr['Close'], 
        "change": ((curr['Close']/prev['Close'])-1)*100,
        "candle": candle_type
    }
    
    # --- [전략 1: 불린저 하단 + 3일 하락 후 장악형 패턴] ---
    # 1. 최근 3일간 종가 하락 (T-3 > T-2 > T-1)
    is_falling = (d3['Close'] > d2['Close']) and (d2['Close'] > prev['Close'])
    # 2. 전일 음봉 & 금일 양봉
    was_bearish = prev['Open'] > prev['Close']
    is_bullish = curr['Close'] > curr['Open']
    # 3. 장악 조건: 금일 종가가 전일 시가 이상
    is_engulfing = curr['Close'] >= prev['Open']
    # 4. 바닥권 조건: 전일 종가가 BB 하단 2% 이내 혹은 금일 저가가 BB 하단 이하(터치)
    is_near_bb_bottom = (prev['Close'] <= prev['BB_L'] * 1.02) or (curr['Low'] <= curr['BB_L'])

    if is_falling and was_bearish and is_bullish and is_engulfing and is_near_bb_bottom:
        return "ENGULFING", res

    # --- 기존 공통 필터 (거래량) ---
    is_korea = ".K" in ticker
    vol_threshold = 1.2 if is_korea else 0.5
    if curr['Volume'] < curr['V_MA20'] * vol_threshold: 
        return None, None

    # --- 기존 전략 매칭 ---
    if curr['MA5'] > curr['MA20'] > curr['MA60'] and curr['Close'] > curr['Tenkan'] and curr['RSI'] > 50:
        return "TREND", res
    if prev['Close'] < prev['BB_L'] * 1.02 and curr['Close'] > curr['Open'] and curr['RSI'] < 50:
        return "REBOUND", res
    if (prev['MA5'] <= prev['MA20'] and curr['MA5'] > curr['MA20']) or \
       (prev['Tenkan'] <= prev['Kijun'] and curr['Tenkan'] > curr['Kijun']):
        return "CROSS", res
            
    return None, None

def process_market(market_name, tickers, names):
    """시장별 프로세스 실행 및 리포트 작성"""
    print(f"[{market_name}] 데이터 수집 및 분석 중...")
    try:
        data = yf.download(tickers, period="8mo", group_by='ticker', threads=True, progress=False)
    except Exception as e:
        print(f"[{market_name}] 다운로드 실패: {e}")
        return
    
    storage = {"TREND": [], "REBOUND": [], "CROSS": [], "ENGULFING": []}
    for ticker in tickers:
        try:
            df = data[ticker].dropna()
            if df.empty: continue
            cat, res = analyze_logic(ticker, df, names[ticker])
            if cat: storage[cat].append(res)
        except: continue

    now_kst = datetime.utcnow() + timedelta(hours=9)
    header = "🇰🇷" if market_name == "KOREA" else "🇺🇸"
    cur_symbol = "₩" if market_name == "KOREA" else "$"
    
    # 1. 메인 전략 리포트 전송
    output = f"{header} **[{market_name} 시장 리포트]**\n"
    output += f"분석: {now_kst.strftime('%m/%d %H:%M')} (KST)\n\n"
    
    sections = [("TREND", "🚀 상승추세"), ("REBOUND", "⚓ 바닥반등"), ("CROSS", "✨ 골든크로스")]
    for key, title in sections:
        output += f"{title}\n"
        items = sorted(storage[key], key=lambda x: abs(x['change']), reverse=True)[:5]
        if not items:
            output += "└ 충족 종목 없음\n"
        else:
            for i in items:
                candle_tag = f" `{i['candle']}`" if i['candle'] else ""
                output += f"└ {i['name']} ({cur_symbol}{i['price']:,.0f}, {i['change']:+.2f}%){candle_tag}\n"
        output += "\n"
    
    send_telegram(output)

    # 2. 불린저 하단 장악형(ENGULFING) 별도 메시지 전송
    if storage["ENGULFING"]:
        eng_msg = f"{header} **[{market_name}] 바닥권 역전환 포착**\n"
        eng_msg += "*(BB하단 근처 + 3일 하락 후 장악형)*\n\n"
        for i in storage["ENGULFING"]:
            candle_tag = f" `{i['candle']}`" if i['candle'] else ""
            eng_msg += f"⭐ **{i['name']}**\n"
            eng_msg += f"└ 현재가: {cur_symbol}{i['price']:,.0f} ({i['change']:+.2f}%)\n"
            eng_msg += f"└ 특이사항: {candle_tag if candle_tag else '바닥 지지 및 장악'}\n\n"
        send_telegram(eng_msg)

def main():
    start_time = datetime.now()
    print(f"--- 분석 시작: {start_time.strftime('%H:%M:%S')} ---")

    # 1. 한국 시장 (시총 상위 300위)
    try:
        kor_listing = fdr.StockListing('KRX').sort_values('Marcap', ascending=False).head(300)
        kor_tickers = [row['Code'] + (".KS" if row['Market'] == 'KOSPI' else ".KQ") for _, row in kor_listing.iterrows()]
        kor_names = dict(zip(kor_tickers, kor_listing['Name']))
        process_market("KOREA", kor_tickers, kor_names)
    except Exception as e:
        print(f"한국 시장 데이터 로드 실패: {e}")

    # 2. 미국 시장 (S&P 500)
    try:
        us_listing = fdr.StockListing('S&P500')
        us_tickers = list(us_listing['Symbol'])
        us_names = dict(zip(us_listing['Symbol'], us_listing['Name']))
        process_market("USA", us_tickers, us_names)
    except Exception as e:
        print(f"미국 시장 데이터 로드 실패: {e}")

    end_time = datetime.now()
    print(f"--- 분석 완료 (소요시간: {end_time - start_time}) ---")

if __name__ == "__main__":
    main()
