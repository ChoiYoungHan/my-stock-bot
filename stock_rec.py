import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import logging
import os

# yfinance 로그 억제
logging.getLogger('yf').setLevel(logging.CRITICAL)

# --- [1. 설정 및 파라미터] ---
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(msg):
    if not (TOKEN and CHAT_ID):
        print("에러: 환경변수(TELEGRAM_TOKEN, CHAT_ID)를 설정해주세요.")
        return
    if not msg: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=20)
    except Exception as e:
        print(f"네트워크 에러: {e}")

def calculate_indicators(df):
    """기술적 지표 계산"""
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    df['MA5'] = c.rolling(5).mean()
    df['MA20'] = c.rolling(20).mean()
    df['MA60'] = c.rolling(60).mean()
    df['V_MA20'] = v.rolling(20).mean()
    
    # 불린저 밴드
    std = c.rolling(20).std()
    df['BB_U'], df['BB_L'] = df['MA20'] + (std * 2), df['MA20'] - (std * 2)
    
    # 일목균형표 전환선
    df['Tenkan'] = (h.rolling(9).max() + l.rolling(9).min()) / 2
    return df

def analyze_logic(ticker, df, name):
    """추세 변곡 및 매매 전략 판별"""
    if len(df) < 30: return None, None
    df = calculate_indicators(df)
    
    try:
        curr = df.iloc[-1]    # 당일 (양봉 기대)
        prev = df.iloc[-2]    # 전일 (도지 기대)
        d2 = df.iloc[-3]      # 2일 전 (하락 흐름 확인)
    except IndexError:
        return None, None

    res = {
        "name": name, "ticker": ticker, "price": curr['Close'], 
        "change": ((curr['Close']/prev['Close'])-1)*100
    }
    
    # 기본 조건: 전일 하락 흐름 및 당일 양봉
    is_prev_falling = d2['Close'] > prev['Close']
    is_today_bullish = curr['Close'] > curr['Open']
    is_near_bottom = (prev['Close'] <= prev['BB_L'] * 1.05) # BB하단 5% 이내
    
    # --- [도지 캔들 상세 판별 로직] ---
    o, h, l, c = prev['Open'], prev['High'], prev['Low'], prev['Close']
    body = abs(c - o)
    upper_tail = h - max(o, c)
    lower_tail = min(o, c) - l
    total_range = h - l
    
    # 1. 몸통이 전체 변동폭의 20% 이하 (매우 작음)
    is_small_body = (body <= total_range * 0.20) if total_range > 0 else False
    # 2. 위꼬리가 몸통보다 길어야 함
    is_upper_tail_long = upper_tail > body
    # 3. 아래꼬리가 몸통보다 길어야 함
    is_lower_tail_long = lower_tail > body
    
    is_true_doji = is_small_body and is_upper_tail_long and is_lower_tail_long

    # --- [전략 적용] ---
    if is_prev_falling and is_true_doji and is_today_bullish and is_near_bottom:
        # 전일 도지가 양봉인 경우 (강력 변곡)
        if prev['Close'] > prev['Open']:
            return "REVERSAL", res
        # 전일 도지가 음봉인 경우 (일반 변곡)
        else:
            return "ANY_DOJI", res

    # --- 공통 필터 및 기타 전략 ---
    if curr['Volume'] < curr['V_MA20'] * 0.4:
        return None, None

    if (df['MA5'].iloc[-2] <= df['MA20'].iloc[-2]) and (df['MA5'].iloc[-1] > df['MA20'].iloc[-1]):
        return "CROSS", res
    if curr['MA5'] > curr['MA20'] > curr['MA60'] and curr['Close'] > curr['Tenkan']:
        return "TREND", res
            
    return None, None

def process_market(market_name, tickers, names):
    print(f"[{market_name}] 분석 시작...")
    try:
        data = yf.download(tickers, period="6mo", group_by='ticker', threads=True, progress=False)
    except Exception as e:
        print(f"다운로드 실패: {e}")
        return
    
    storage = {"TREND": [], "CROSS": [], "REVERSAL": [], "ANY_DOJI": []}
    for ticker in tickers:
        try:
            df = data[ticker].dropna() if len(tickers) > 1 else data.dropna()
            if df.empty or len(df) < 15: continue
            cat, res = analyze_logic(ticker, df, names[ticker])
            if cat: storage[cat].append(res)
        except: continue

    now_kst = datetime.utcnow() + timedelta(hours=9)
    header = "🇰🇷" if market_name == "KOREA" else "🇺🇸"
    cur_symbol = "₩" if market_name == "KOREA" else "$"
    
    # 1. 통합 리포트
    output = f"{header} **[{market_name} 시장 리포트]**\n분석: {now_kst.strftime('%m/%d %H:%M')}\n\n"
    for key, title in [("TREND", "🚀 상승추세"), ("CROSS", "✨ 골든크로스")]:
        output += f"{title}\n"
        items = sorted(storage[key], key=lambda x: abs(x['change']), reverse=True)[:5]
        output += ("\n".join([f"└ {i['name']} ({cur_symbol}{i['price']:,.0f}, {i['change']:+.2f}%)" for i in items]) if items else "└ 없음") + "\n\n"
    send_telegram(output)

    # 2. 변곡점 리포트 (꼬리 조건 강화 버전)
    rev_found = False
    rev_msg = f"{header} **[{market_name}] 꼬리 긴 도지 변곡**\n"
    
    combined_rev = storage["REVERSAL"] + storage["ANY_DOJI"]
    if combined_rev:
        rev_found = True
        for i in combined_rev:
            tag = "🔥양봉도지" if i in storage["REVERSAL"] else "⚖️음봉도지"
            rev_msg += f"└ {i['name']} ({cur_symbol}{i['price']:,.0f}) [{tag}]\n"

    if rev_found:
        send_telegram(rev_msg)
    else:
        send_telegram(f"{header} **[{market_name}]** 꼬리 긴 도지 반등 종목이 없습니다.")

def main():
    try:
        kor_listing = fdr.StockListing('KRX').sort_values('Marcap', ascending=False).head(300)
        kor_tickers = [row['Code'] + (".KS" if row['Market'] == 'KOSPI' else ".KQ") for _, row in kor_listing.iterrows()]
        process_market("KOREA", kor_tickers, dict(zip(kor_tickers, kor_listing['Name'])))
    except Exception as e: print(f"한국 오류: {e}")

    try:
        us_listing = fdr.StockListing('S&P500')
        us_tickers = [t.replace('.', '-') for t in us_listing['Symbol']]
        process_market("USA", us_tickers, dict(zip(us_tickers, us_listing['Name'])))
    except Exception as e: print(f"미국 오류: {e}")

if __name__ == "__main__":
    main()
