import os
import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import logging

# yfinance 내부 로그 억제
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# --- [1. 환경 변수 관리] ---
# GitHub Secrets에 저장한 이름과 반드시 일치해야 합니다.
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(msg):
    if not msg: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print(f"텔레그램 전송 실패: {e}")

def calculate_indicators(df):
    """기술적 지표 계산 로직"""
    # 데이터가 부족할 경우 에러 방지
    if len(df) < 60: return df
    
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    
    # 이동평균선
    df['MA5'] = c.rolling(5).mean()
    df['MA20'] = c.rolling(20).mean()
    df['MA60'] = c.rolling(60).mean()
    df['V_MA20'] = v.rolling(20).mean() # 거래량 이평
    
    # 볼린저 밴드
    std = c.rolling(20).std()
    df['BB_U'], df['BB_L'] = df['MA20'] + (std * 2), df['MA20'] - (std * 2)
    
    # RSI
    delta = c.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0], down[down > 0] = 0, 0
    df['RSI'] = 100 - (100 / (1 + (up.ewm(13).mean() / down.abs().ewm(13).mean())))
    
    # 일목균형표 전환선/기준선
    df['Tenkan'] = (h.rolling(9).max() + l.rolling(9).min()) / 2
    df['Kijun'] = (h.rolling(26).max() + l.rolling(26).min()) / 2
    return df

def analyze_logic(ticker, df, name):
    """종목별 전략 판별 로직"""
    if len(df) < 60: return None, None
    
    df = calculate_indicators(df)
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 거래량 필터 (최근 20일 평균 대비 1.2배 이상)
    if curr['Volume'] < curr['V_MA20'] * 1.2: 
        return None, None

    res = {
        "name": name, 
        "ticker": ticker, 
        "price": curr['Close'], 
        "change": ((curr['Close'] / prev['Close']) - 1) * 100
    }
    
    # 1. 상승추세: 정배열 + 일목호전 + RSI 강세
    if curr['MA5'] > curr['MA20'] > curr['MA60'] and curr['Close'] > curr['Tenkan'] and curr['RSI'] > 50:
        return "TREND", res
        
    # 2. 바닥반등: 볼린저 하단 인접 후 양봉 반등
    if prev['Close'] < prev['BB_L'] * 1.02 and curr['Close'] > curr['Open'] and curr['RSI'] < 50:
        return "REBOUND", res
        
    # 3. 골든크로스: 이평선 교차 혹은 일목 전환/기준선 교차
    if (prev['MA5'] <= prev['MA20'] and curr['MA5'] > curr['MA20']) or \
       (prev['Tenkan'] <= prev['Kijun'] and curr['Tenkan'] > curr['Kijun']):
        return "CROSS", res
            
    return None, None

def process_market(market_name, tickers, names):
    """시장별 데이터 수집 및 리포트 전송"""
    print(f"[{market_name}] 분석 시작...")
    
    # 데이터 다운로드
    try:
        data = yf.download(tickers, period="8mo", group_by='ticker', threads=True, progress=False)
    except Exception as e:
        print(f"데이터 다운로드 에러: {e}")
        return

    storage = {"TREND": [], "REBOUND": [], "CROSS": []}
    
    for ticker in tickers:
        try:
            # yfinance 멀티데이터 인덱싱 대응
            df = data[ticker].dropna()
            if df.empty: continue
            
            cat, res = analyze_logic(ticker, df, names[ticker])
            if cat: 
                storage[cat].append(res)
        except:
            continue

    # 한국 시간(KST) 계산 (UTC+9)
    now_kst = datetime.utcnow() + timedelta(hours=9)
    
    header = "🇰🇷" if market_name == "KOREA" else "🇺🇸"
    output = f"{header} **[{market_name} 시장 분석 리포트]**\n"
    output += f"분석 시간: {now_kst.strftime('%Y-%m-%d %H:%M')} (KST)\n"
    output += "_거래량 동반 종목 선별_\n\n"
    
    sections = [("TREND", "🚀 상승추세"), ("REBOUND", "⚓ 바닥반등"), ("CROSS", "✨ 골든크로스")]
    
    for key, title in sections:
        output += f"{title}\n"
        # 변동성(절대값)이 큰 순서대로 5개 추출
        items = sorted(storage[key], key=lambda x: abs(x['change']), reverse=True)[:5]
        
        if not items:
            output += "└ 충족 종목 없음\n"
        else:
            for i in items:
                cur = "₩" if market_name == "KOREA" else "$"
                output += f"└ {i['name']} ({cur}{i['price']:,.2f}, {i['change']:+.2f}%)\n"
        output += "\n"
    
    send_telegram(output)

def main():
    # 1. 한국 시장 설정 (시총 상위 300)
    try:
        kor_listing = fdr.StockListing('KRX').sort_values('Marcap', ascending=False).head(300)
        kor_tickers = [row['Code'] + (".KS" if row['Market'] == 'KOSPI' else ".KQ") for _, row in kor_listing.iterrows()]
        kor_names = dict(zip(kor_tickers, kor_listing['Name']))
        process_market("KOREA", kor_tickers, kor_names)
    except Exception as e:
        print(f"한국 시장 처리 중 에러: {e}")

    # 2. 미국 시장 설정 (S&P 500)
    try:
        us_listing = fdr.StockListing('S&P500')
        us_tickers = list(us_listing['Symbol'])
        us_names = dict(zip(us_listing['Symbol'], us_listing['Name']))
        process_market("USA", us_tickers, us_names)
    except Exception as e:
        print(f"미국 시장 처리 중 에러: {e}")

if __name__ == "__main__":
    main()
