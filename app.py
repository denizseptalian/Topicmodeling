# ================= IMPORT =================
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from wordcloud import WordCloud
from collections import Counter
import re, feedparser, urllib.parse

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# OPTIONAL ALPHA
try:
    from alpha_vantage.timeseries import TimeSeries
    ALPHA_OK = True
except:
    ALPHA_OK = False

# ================= CONFIG =================
AV_API_KEY = "YQNUKAH419JA2RYV"
HEADERS = {"User-Agent": "Mozilla/5.0"}

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

st.set_page_config(layout="wide")

# ================= SESSION =================
if "run" not in st.session_state:
    st.session_state.run = False

# ================= INIT VAR =================
df_full = None
df_range = None
df_news = None

# ================= SMART KEYWORD =================
def smart_keyword(keyword, ticker):
    if not keyword or keyword.strip()=="":
        keyword = ticker

    keyword = str(keyword).replace("\n"," ").replace("\r"," ")
    keyword = " ".join(keyword.split())

    suggestions = [
        keyword,
        f"saham {keyword}",
        f"{keyword} Indonesia",
        f"{keyword} berita",
        f"{keyword} stock"
    ]

    return keyword, suggestions, urllib.parse.quote(keyword)

# ================= KURS =================
@st.cache_data(ttl=3600)
def get_kurs():
    try:
        r = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=IDR", timeout=10).json()
        return r['rates']['IDR']
    except:
        return 15500

# ================= PREPROCESS =================
def preprocess_text(text):
    if pd.isna(text): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return stopword.remove(text)

# ================= SENTIMENT =================
pos_words = {"naik","laba","untung","positif","menguat","tumbuh"}
neg_words = {"turun","rugi","anjlok","negatif","melemah","buruk"}

def sentiment_score(text):
    s=0
    for w in text.split():
        if w in pos_words: s+=1
        if w in neg_words: s-=1
    return s

def sentiment_label(s):
    return "Positif" if s>0 else "Negatif" if s<0 else "Netral"

# ================= YAHOO =================
@st.cache_data(ttl=600)
def get_yahoo(symbol, start, end, is_indo):

    def to_unix(d):
        return int(datetime(d.year,d.month,d.day,tzinfo=timezone.utc).timestamp())

    sym = f"{symbol}.JK" if is_indo else symbol
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"

    r = requests.get(url, headers=HEADERS, params={
        "interval":"1d",
        "period1":to_unix(start),
        "period2":to_unix(end+timedelta(days=1))
    }, timeout=10)

    data = r.json()
    result = data.get("chart",{}).get("result")
    if not result: return None

    ts = result[0]["timestamp"]
    close = result[0]["indicators"]["quote"][0]["close"]

    df = pd.DataFrame({
        "Date":[datetime.fromtimestamp(t).date() for t in ts],
        "Close":close
    }).dropna()

    df['Prev_Close'] = df['Close'].shift(1)
    df['Price_Change'] = df['Close'] - df['Prev_Close']
    df['Pct_Change (%)'] = (df['Price_Change']/df['Prev_Close'])*100

    if not is_indo:
        df['Close_IDR'] = df['Close'] * get_kurs()

    return df

def get_full_history(symbol, is_indo):
    return get_yahoo(symbol, datetime.now()-timedelta(days=365*3), datetime.now(), is_indo)

# ================= NEWS =================
@st.cache_data(ttl=600)
def get_news(keyword_encoded, start, end):

    data=[]
    for d in pd.date_range(start,end):
        url=f"https://news.google.com/rss/search?q={keyword_encoded}+after:{d.date()}+before:{(d+timedelta(days=1)).date()}&hl=id&gl=ID&ceid=ID:id"
        feed=feedparser.parse(url)

        for e in feed.entries:
            data.append({
                "Date": pd.to_datetime(e.published).date(),
                "title": e.title,
                "desc": getattr(e,"summary",""),
                "media": e.source.title if hasattr(e,"source") else ""
            })

    if not data: return None,None,None,None,None

    df=pd.DataFrame(data).drop_duplicates(subset="title")
    df["doc"]=(df["title"]+" "+df["desc"]).apply(preprocess_text)
    df["sentiment_score"]=df["doc"].apply(sentiment_score)
    df["sentiment_label"]=df["sentiment_score"].apply(sentiment_label)

    text=" ".join(df["doc"])
    wc = WordCloud(background_color="white").generate(text) if text.strip() else None
    common = Counter(text.split()).most_common(10) if text.strip() else None

    media = df["media"].value_counts().head(10)

    df_daily = df.groupby("Date")["title"].apply(lambda x:" | ".join(x)).reset_index()
    sent = df.groupby("Date")["sentiment_score"].mean().reset_index()
    df_daily = pd.merge(df_daily, sent, on="Date", how="left")

    return df, df_daily, wc, common, media

# ================= LSTM PREDIKSI BESOK =================
def predict_next_day(df):
    if df is None or len(df) < 20:
        return None

    data = df[['Close']].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X = []
    for i in range(10, len(data_scaled)):
        X.append(data_scaled[i-10:i])
    X = np.array(X)

    y = data_scaled[10:]

    model = Sequential([
        LSTM(50, input_shape=(10,1)),
        Dense(1)
    ])

    model.compile("adam","mse")
    model.fit(X, y, epochs=5, verbose=0)

    last_seq = data_scaled[-10:]
    last_seq = np.reshape(last_seq, (1,10,1))

    pred = model.predict(last_seq)
    pred = scaler.inverse_transform(pred)

    return float(pred[0][0])

# ================= AUTO SIGNAL =================
def auto_signal(df, next_price):
    if df is None or next_price is None:
        return "NO DATA", 0

    df = df.dropna(subset=['sentiment_score','Pct_Change (%)'])
    if len(df) < 5:
        return "DATA KURANG", 0

    last = df.tail(5)

    avg_sent = last['sentiment_score'].mean()
    momentum = last['Pct_Change (%)'].mean()
    last_price = df['Close'].iloc[-1]

    expected = (next_price - last_price)/last_price

    score = 0
    if avg_sent > 0: score += 1
    if momentum > 0: score += 1
    if expected > 0: score += 2
    if expected > 0.02: score += 1

    if score >= 4: signal = "🔥 STRONG BUY"
    elif score >= 3: signal = "✅ BUY"
    elif score == 2: signal = "⚖️ HOLD"
    else: signal = "❌ SELL"

    return signal, min(score/5*100,100)

# ================= UI =================
st.title("💹 Analisis Sentimen + AI Trading Signal")

market = st.selectbox("Market", ["Indonesia","Global"])
ticker = st.text_input("Ticker","BBCA")
keyword_input = st.text_input("Keyword Berita")

kw, suggestions, kw_encoded = smart_keyword(keyword_input, ticker)
st.write(suggestions)

start = st.date_input("Start", datetime.now()-timedelta(days=30))
end = st.date_input("End", datetime.now())

if st.button("RUN"):
    st.session_state.run = True

if st.session_state.run:

    with st.spinner("Loading data..."):

        is_indo = market=="Indonesia"

        df_full = get_full_history(ticker,is_indo)
        df_range = get_yahoo(ticker,start,end,is_indo)
        df_news, df_daily, wc, common, media = get_news(kw_encoded,start,end)

        if df_daily is not None and df_range is not None:
            df_range = pd.merge(df_range, df_daily, on="Date", how="left")

    # ================= PREDIKSI =================
    st.subheader("🔮 Prediksi Besok")

    next_price = predict_next_day(df_full)

    if next_price:
        last_price = df_full['Close'].iloc[-1]
        change = ((next_price-last_price)/last_price)*100

        st.metric("Prediksi", f"{next_price:.2f}", f"{change:.2f}%")

    # ================= SIGNAL =================
    st.subheader("🚦 Signal")

    signal, conf = auto_signal(df_range, next_price)

    if "BUY" in signal:
        st.success(f"{signal} ({conf:.1f}%)")
    elif "SELL" in signal:
        st.error(f"{signal} ({conf:.1f}%)")
    else:
        st.warning(f"{signal} ({conf:.1f}%)")
