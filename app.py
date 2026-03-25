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

# ================= SESSION STATE =================
if "run" not in st.session_state:
    st.session_state.run = False

# ================= SMART KEYWORD =================
def smart_keyword(keyword, Ticker):
    if not keyword or keyword.strip()=="":
        keyword = Ticker

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

# ================= YAHOO SAFE =================
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

# ================= FULL HISTORICAL =================
def get_full_history(symbol, is_indo):
    return get_yahoo(symbol, datetime.now()-timedelta(days=365*3), datetime.now(), is_indo)

# ================= ALPHA =================
@st.cache_data(ttl=600)
def get_alpha(symbol, start, end):

    if not ALPHA_OK: return None

    try:
        ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
        data,_ = ts.get_daily(symbol=symbol)

        df = data.rename(columns={'4. close':'Close'})
        df.index = pd.to_datetime(df.index)

        df = df.loc[(df.index>=pd.to_datetime(start)) & (df.index<=pd.to_datetime(end))]
        df = df.reset_index().rename(columns={"index":"Date"})
        df['Date'] = pd.to_datetime(df['Date']).dt.date

        df['Prev_Close'] = df['Close'].shift(1)
        df['Price_Change'] = df['Close'] - df['Prev_Close']
        df['Pct_Change (%)'] = (df['Price_Change']/df['Prev_Close'])*100
        df['Close_IDR'] = df['Close'] * get_kurs()

        return df

    except:
        return None

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

    wc, common = None, None
    if text.strip():
        wc = WordCloud(background_color="white").generate(text)
        common = Counter(text.split()).most_common(10)

    media = df["media"].value_counts().head(10)

    df_daily = df.groupby("Date")["title"].apply(lambda x:" | ".join(x)).reset_index()
    sent = df.groupby("Date")["sentiment_score"].mean().reset_index()

    df_daily = pd.merge(df_daily, sent, on="Date", how="left")

    return df, df_daily, wc, common, media

# ================= LSTM =================
def lstm(df):
    data=df[['Close']].values
    if len(data)<20: return None

    scaler=MinMaxScaler()
    data=scaler.fit_transform(data)

    X,y=[],[]
    for i in range(10,len(data)):
        X.append(data[i-10:i])
        y.append(data[i])

    X,y=np.array(X),np.array(y)

    model=Sequential([
        LSTM(50,input_shape=(10,1)),
        Dense(1)
    ])

    model.compile("adam","mse")
    model.fit(X,y,epochs=5,verbose=0)

    pred=model.predict(X)
    pred=scaler.inverse_transform(pred)

    df_out=df.iloc[10:].copy()
    df_out["Prediksi"]=pred

    return df_out

# ================= STYLE =================
def color(val):
    if pd.isna(val): return ""
    return "color:green" if val>0 else "color:red"

# ================= UI =================
st.title("💹 Analisis Sentimen Berita Ekonomi pada Google News dan Pengaruhnya terhadap Volatilitas serta Pergerakan Intraday Harga Saham")

market = st.selectbox("Market", ["Indonesia","Global"])
source = st.selectbox("Data Source", ["Yahoo","Alpha"])
ticker = st.text_input("Ticker","BBCA")
keyword_input = st.text_input("Keyword Berita")

kw, suggestions, kw_encoded = smart_keyword(keyword_input, ticker)

st.caption("💡 Smart Keyword:")
st.write(suggestions)

start = st.date_input("Start", datetime.now()-timedelta(days=30))
end = st.date_input("End", datetime.now())

if st.button("RUN"):
    st.session_state.run = True

if st.session_state.run:

    with st.spinner("🔄 Mengambil data, mohon tunggu..."):

        is_indo = market=="Indonesia"

        df_full = get_full_history(ticker,is_indo)
        df_range = get_yahoo(ticker,start,end,is_indo) if source=="Yahoo" else get_alpha(ticker,start,end)
        df_news, df_daily, wc, common, media = get_news(kw_encoded,start,end)

        if df_daily is not None and df_range is not None:
            df_range = pd.merge(df_range, df_daily, on="Date", how="left")

    # ================= KORELASI =================
    st.subheader("📊 Korelasi Sentimen vs Perubahan Harga")

    if df_range is not None and 'sentiment_score' in df_range.columns:

        df_corr = df_range[['Date','Pct_Change (%)','sentiment_score']].dropna()

        if len(df_corr) > 2:
            corr = df_corr['Pct_Change (%)'].corr(df_corr['sentiment_score'])

            st.metric("Nilai Korelasi", f"{corr:.4f}")

        else:
            st.warning("Data tidak cukup")

    # ================= REKOMENDASI =================
    st.subheader("💡 Rekomendasi Trading")

    def rekomendasi(df):
        df = df.dropna(subset=['sentiment_score','Pct_Change (%)'])
        last = df.tail(3)

        avg_sent = last['sentiment_score'].mean()
        avg_return = last['Pct_Change (%)'].mean()

        if avg_sent > 0.5 and avg_return > 0:
            return "BELI"
        elif avg_sent < 0 and avg_return < 0:
            return "JANGAN BELI"
        else:
            return "PANTAU"

    if df_range is not None:
        st.write("Hari ini:", rekomendasi(df_range))
