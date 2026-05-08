import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import logging
import os

# yfinance 불필요한 로그 출력 억제
logging.getLogger('yf').setLevel(logging.CRITICAL)

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
    """봉(Candle)의 모양 분석"""
    o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
    total = h - l
    if total <= 0: return ""
    
    body = abs(c - o)
    lower_tail = min(o, c) - l
    
    # 도지형 판단 (몸통이 전체 변동폭의 15% 이하)
    if body < total * 0.15: return "⚖️도지(변곡)"
    if lower_tail > total * 0.5: return "⚓아래꼬리(지지)"
    if c > o and body > total * 0.8: return "🔥장대양봉"
    
    return ""

def analyze_logic(ticker, df, name):
    """전략 판별 로직"""
    if len(df) < 30: return None, None
    df = calculate_indicators(df)
    
    try:
        curr = df.iloc[-1]    # 당일 (양봉 반등 기대)
        prev = df.iloc[-2]    # 전일 (양봉 도지 기대)
        d2 = df.iloc[-3]      # 2일 전 (하락)
        d3 = df.iloc[-4]      # 3일 전 (하락 시작)
    except IndexError:
        return None, None
    
    candle_type = analyze_candle(curr)
    res = {
        "name": name, 
        "ticker": ticker, 
        "price": curr['Close'], 
        "change": ((curr['Close']/prev['Close'])-1)*100,
        "candle": candle_type
    }
    
    # --- [전략 1: 변곡점 - 하락 -> 전일 종가 아래 '양봉 도지' -> 당일 양봉] ---
    is_falling = (d3['Close'] > d2['Close'])
    
    prev_total = (prev['High'] - prev['Low'])
    # 몸통이 양봉(c > o)이면서 아주 작아야 함(0 < body <= 15%)
    is_prev_positive_doji = (prev['Close'] > prev['Open']) and \
                             ((prev['Close'] - prev['Open']) <= prev_total * 0.15 if prev_total > 0 else False)
    # 도지의 고가가 전일 종가보다 낮아야 함 (하락 갭 또는 저점 갱신)
    is_doji_below = prev['High'] < d2['Close']
    
    is_today_bullish = curr['Close'] > curr['Open']
    is_near_bb_bottom = (prev['Close'] <= prev['BB_L'] * 1.03) or (curr['Low'] <= curr['BB_L'] * 1.02)

    if is_falling and is_prev_positive_doji and is_doji_below and is_today_bullish and is_near_bb_bottom:
        return "REVERSAL", res

    # --- 공통 필터 ---
    if curr['Volume'] < curr['V_MA20'] * 0.5: 
        return None, None

    # --- [전략 2: 골든크로스] ---
    if (df['MA5'].iloc[-2] <= df['MA20'].iloc[-2]) and (df['MA5'].iloc[-1] > df['MA20'].iloc[-1]):
        return "CROSS", res

    # --- [전략 3: 상승추세 및 바닥반등] ---
    if curr['MA5'] > curr['MA20'] > curr['MA60'] and curr['Close'] > curr['Tenkan']:
        return "TREND", res
    if prev['Close'] < prev['BB_L'] * 1.02 and curr['Close'] > curr['Open']:
        return "REBOUND", res
            
    return None, None

def process_market(market_name, tickers, names):
    """시장별 프로세스 실행"""
    print(f"[{market_name}] 분석 중... (대상: {len(tickers)}개)")
    
    try:
        data = yf.download(tickers, period="6mo", group_by='ticker', threads=True, progress=False, timeout=40)
    except Exception as e:
        print(f"[{market_name}] 다운로드 오류: {e}")
        return
    
    storage = {"TREND": [], "REBOUND": [], "CROSS": [], "REVERSAL": []}
    
    for ticker in tickers:
        try:
            df = data[ticker].dropna() if len(tickers) > 1 else data.dropna()
            if df.empty or len(df) < 25: continue
            
            cat, res = analyze_logic(ticker, df, names[ticker])
            if cat: storage[cat].append(res)
        except:
            continue

    now_kst = datetime.utcnow() + timedelta(hours=9)
    header = "🇰🇷" if market_name == "KOREA" else "🇺🇸"
    cur_symbol = "₩" if market_name == "KOREA" else "$"
    
    # 1. 메인 리포트 (추세, 반등, 골든크로스 포함)
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
                output += f"└ {i['name']} ({cur_symbol}{i['price']:,.2f}, {i['change']:+.2f}%)\n"
        output += "\n"
    send_telegram(output)

    # 2. 변곡점 리포트 (하락 갭 양봉도지)
    if storage["REVERSAL"]:
        rev_msg = f"{header} **[{market_name}] 양봉도지 반등 포착**\n"
        rev_msg += "*(하락 후 저가 양봉도지 + 금일 양봉)*\n\n"
        for i in storage["REVERSAL"]:
            rev_msg += f"✨ **{i['name']}**\n"
            rev_msg += f"└ 현재가: {cur_symbol}{i['price']:,.2f} ({i['change']:+.2f}%)\n"
            rev_msg += f"└ 특징: 이전 종가 아래에서 양봉도지 변곡성공\n\n"
        send_telegram(rev_msg)

def main():
    # 1. 한국 시장
    try:
        kor_listing = fdr.StockListing('KRX').sort_values('Marcap', ascending=False).head(300)
        kor_tickers = [row['Code'] + (".KS" if row['Market'] == 'KOSPI' else ".KQ") for _, row in kor_listing.iterrows()]
        kor_names = dict(zip(kor_tickers, kor_listing['Name']))
        process_market("KOREA", kor_tickers, kor_names)
    except Exception as e:
        print(f"한국 분석 오류: {e}")

    # 2. 미국 시장
    try:
        us_listing = fdr.StockListing('S&P500')
        us_tickers = [t.replace('.', '-') for t in us_listing['Symbol']]
        us_names = dict(zip(us_tickers, us_listing['Symbol'])) # 이름을 티커로 대체하거나 매핑 수정
        process_market("USA", us_tickers, us_names)
    except Exception as e:
        print(f"미국 분석 오류: {e}")

if __name__ == "__main__":
    main()
