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
    """텔레그램 메시지 전송 함수"""
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
    
    # 불린저 밴드
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
    """봉(Candle)의 모양 분석 로직"""
    o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
    total = h - l
    if total == 0: return ""
    
    body = abs(c - o)
    upper_tail = h - max(o, c)
    lower_tail = min(o, c) - l
    
    if body < total * 0.1: return "⚖️도지(변곡)"
    if lower_tail > total * 0.5: return "⚓아래꼬리(지지)"
    if upper_tail > total * 0.5: return "⚠️위꼬리(저항)"
    if c > o and body > total * 0.8: return "🔥장대양봉"
    
    return ""

def analyze_logic(ticker, df, name):
    """전략 판별 로직"""
    if len(df) < 60: return None, None
    df = calculate_indicators(df)
    
    curr = df.iloc[-1]   # 당일
    prev = df.iloc[-2]   # 전일 (도지 기대)
    d2 = df.iloc[-3]     # 2일 전 (하락 기대)
    d3 = df.iloc[-4]     # 3일 전 (하락 기대)
    
    candle_type = analyze_candle(curr)

    res = {
        "name": name, 
        "ticker": ticker, 
        "price": curr['Close'], 
        "change": ((curr['Close']/prev['Close'])-1)*100,
        "candle": candle_type
    }
    
    # --- [신규 전략: 2일 하락 + 도지 + 당일 양봉 + BB바닥] ---
    # 1. 2일 연속 하락 (T-3 > T-2)
    is_falling = (d3['Close'] > d2['Close']) and (d2['Close'] > prev['Close'])
    # 2. 전일(T-1)이 도지형 캔들 (몸통이 전체 변동폭의 15% 미만)
    prev_total = (prev['High'] - prev['Low'])
    is_prev_doji = (abs(prev['Open'] - prev['Close']) < prev_total * 0.15) if prev_total > 0 else False
    # 3. 금일(T)이 양봉
    is_today_bullish = curr['Close'] > curr['Open']
    # 4. 바닥권 조건: 전일이나 금일이 BB 하단 근처 (2% 이내)
    is_near_bb_bottom = (prev['Close'] <= prev['BB_L'] * 1.02) or (curr['Low'] <= curr['BB_L'] * 1.01)

    if is_falling and is_prev_doji and is_today_bullish and is_near_bb_bottom:
        return "REVERSAL", res

    # --- 기존 공통 필터 (거래량) ---
    is_korea = ".K" in ticker
    vol_threshold = 1.1 if is_korea else 0.5
    if curr['Volume'] < curr['V_MA20'] * vol_threshold: 
        return None, None

    # --- 기존 전략 매칭 ---
    if curr['MA5'] > curr['MA20'] > curr['MA60'] and curr['Close'] > curr['Tenkan'] and curr['RSI'] > 50:
        return "TREND", res
    if prev['Close'] < prev['BB_L'] * 1.02 and curr['Close'] > curr['Open'] and curr['RSI'] < 50:
        return "REBOUND", res
    if (prev['MA5'] <= prev['MA20'] and curr['MA5'] > prev['MA20']):
        return "CROSS", res
            
    return None, None

def process_market(market_name, tickers, names):
    """시장별 프로세스 실행"""
    print(f"[{market_name}] 데이터 분석 중...")
    try:
        data = yf.download(tickers, period="8mo", group_by='ticker', threads=True, progress=False)
    except:
        return
    
    storage = {"TREND": [], "REBOUND": [], "CROSS": [], "REVERSAL": []}
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
    
    # 1. 일반 리포트
    output = f"{header} **[{market_name} 시장 리포트]**\n"
    output += f"분석: {now_kst.strftime('%m/%d %H:%M')} (KST)\n\n"
    
    sections = [("TREND", "🚀 상승추세"), ("REBOUND", "⚓ 바닥반등"), ("CROSS", "✨ 골든크로스")]
    for key, title in sections:
        output += f"{title}\n"
        items = sorted(storage[key], key=lambda x: abs(x['change']), reverse=True)[:5]
        if not items:
            output += "└ 없음\n"
        else:
            for i in items:
                output += f"└ {i['name']} ({cur_symbol}{i['price']:,.0f}, {i['change']:+.2f}%)\n"
        output += "\n"
    send_telegram(output)

    # 2. 변곡점(REVERSAL) 리포트 별도 전송
    if storage["REVERSAL"]:
        rev_msg = f"{header} **[{market_name}] 변곡점 포착 (하락-도지-양봉)**\n"
        rev_msg += "*(BB하단 지지 + 추세 전환 신호)*\n\n"
        for i in storage["REVERSAL"]:
            candle_tag = f" `{i['candle']}`" if i['candle'] else ""
            rev_msg += f"💎 **{i['name']}**\n"
            rev_msg += f"└ 현재가: {cur_symbol}{i['price']:,.0f} ({i['change']:+.2f}%){candle_tag}\n\n"
        send_telegram(rev_msg)

def main():
    print("--- 분석 시작 ---")
    # 한국 시장
    try:
        kor_listing = fdr.StockListing('KRX').sort_values('Marcap', ascending=False).head(300)
        kor_tickers = [row['Code'] + (".KS" if row['Market'] == 'KOSPI' else ".KQ") for _, row in kor_listing.iterrows()]
        kor_names = dict(zip(kor_tickers, kor_listing['Name']))
        process_market("KOREA", kor_tickers, kor_names)
    except: pass

    # 미국 시장
    try:
        us_listing = fdr.StockListing('S&P500')
        us_tickers = list(us_listing['Symbol'])
        us_names = dict(zip(us_listing['Symbol'], us_listing['Name']))
        process_market("USA", us_tickers, us_names)
    except: pass
    print("--- 분석 완료 ---")

if __name__ == "__main__":
    main()
