import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import streamlit as st
from wordcloud import WordCloud
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
from datetime import datetime, timedelta
import feedparser
import time
import requests

# ================= CONFIG =================
AV_API_KEY = "ISI_API_KEY_KAMU"

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

st.set_page_config(layout="wide")

# ================= KURS =================
@st.cache_data(ttl=3600)
def get_kurs():
    try:
        url = "https://api.exchangerate.host/latest?base=USD&symbols=IDR"
        res = requests.get(url).json()
        return res['rates']['IDR']
    except:
        return 15500

# ================= PREPROCESS =================
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return stopword.remove(text)

# ================= NORMALIZE =================
def normalize_df(df):
    if df is None or df.empty:
        return None
    df = df.reset_index()
    if 'Date' not in df.columns:
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

# ================= YAHOO =================
@st.cache_data(ttl=300)
def get_yahoo(ticker, start, end):
    try:
        ticker_jk = ticker if ticker.endswith(".JK") else ticker + ".JK"
        time.sleep(1)
        df = yf.Ticker(ticker_jk).history(start=start, end=end)
        return normalize_df(df)
    except:
        return None

# ================= ALPHA =================
@st.cache_data(ttl=300)
def get_alpha(ticker, start, end):
    try:
        ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=ticker, outputsize='compact')

        df = data.rename(columns={'4. close': 'Close'})
        df.index = pd.to_datetime(df.index)

        df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        return normalize_df(df)
    except:
        return None

# ================= MULTI SOURCE =================
def get_stock(source, ticker, start, end):
    if source == "Yahoo Finance":
        df = get_yahoo(ticker, start, end)
        if df is None:
            st.warning("Yahoo gagal → Alpha")
            df = get_alpha(ticker, start, end)
    else:
        df = get_alpha(ticker, start, end)
        if df is None:
            st.warning("Alpha gagal → Yahoo")
            df = get_yahoo(ticker, start, end)

    if df is None:
        return None

    kurs = get_kurs()

    df['Prev_Close'] = df['Close'].shift(1)
    df['Price_Change'] = df['Close'] - df['Prev_Close']
    df['Pct_Change (%)'] = (df['Price_Change'] / df['Prev_Close']) * 100
    df['Close_IDR'] = df['Close'] * kurs

    return df

# ================= NEWS =================
@st.cache_data(ttl=300)
def get_news(keyword):
    url = f"https://news.google.com/rss/search?q={keyword.replace(' ', '%20')}&hl=id&gl=ID&ceid=ID:id"
    feed = feedparser.parse(url)

    if not feed.entries:
        return None, None, None, None

    data = []
    for e in feed.entries:
        data.append({
            "Date": pd.to_datetime(e.get("published")).date(),
            "title": e.title,
            "source": e.get("source", {}).get("title", "")
        })

    df = pd.DataFrame(data)

    # GROUP BERITA PER HARI
    df_daily = df.groupby("Date")["title"].apply(lambda x: " | ".join(x)).reset_index()

    # WORDCLOUD
    df["clean"] = df["title"].apply(preprocess_text)
    text = " ".join(df["clean"])

    wc, common = None, None
    if text.strip():
        wc = WordCloud(background_color="white").generate(text)
        common = Counter(text.split()).most_common(10)

    media = df["source"].value_counts().head(10)

    return df_daily, wc, common, media

# ================= STYLE =================
def color_change(val):
    if pd.isna(val):
        return ""
    return "color: green" if val > 0 else "color: red"

# ================= UI =================
st.title("💹 Analisis Saham + Berita (MERGE HARIAN)")

source = st.sidebar.selectbox("Sumber Saham", ["Yahoo Finance", "Alpha Vantage"])
ticker = st.sidebar.text_input("Ticker", "BBCA")
keyword = st.sidebar.text_input("Keyword", "Bank BCA")

c1, c2 = st.sidebar.columns(2)
start = c1.date_input("Mulai", datetime.now() - timedelta(days=30))
end = c2.date_input("Selesai", datetime.now())

if st.sidebar.button("🚀 Jalankan"):

    df_s = get_stock(source, ticker, start, end)
    df_n, wc, common, media = get_news(keyword)

    if df_s is not None and df_n is not None:

        # ================= MERGE =================
        df_merge = pd.merge(df_s, df_n, on="Date", how="left")

        st.subheader("📊 Data Gabungan Saham + Berita")

        cols = [
            'Date','Prev_Close','Close','Close_IDR',
            'Price_Change','Pct_Change (%)','title'
        ]
        cols = [c for c in cols if c in df_merge.columns]

        styled = df_merge[cols].style.format({
            "Prev_Close": "{:.2f}",
            "Close": "{:.2f}",
            "Close_IDR": "Rp {:,.0f}",
            "Price_Change": "{:.2f}",
            "Pct_Change (%)": "{:.2f}"
        }).applymap(color_change, subset=['Price_Change','Pct_Change (%)'])

        st.dataframe(styled, use_container_width=True)

        # ================= GRAFIK =================
        st.subheader("📈 Grafik Harga (IDR)")
        st.line_chart(df_merge.set_index("Date")["Close_IDR"])

        # ================= WORDCLOUD =================
        if wc:
            st.subheader("☁️ WordCloud")
            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)

        if common:
            st.subheader("📌 Top Kata")
            st.table(pd.DataFrame(common, columns=["Kata","Jumlah"]))

        if not media.empty:
            st.subheader("🏆 Top Media")
            st.table(
                media.reset_index()
                .rename(columns={"index":"Media", "source":"Jumlah"})
            )

    else:
        st.error("❌ Data tidak lengkap (saham / berita gagal)")
