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

# =========================
# CONFIG
# =========================
AV_API_KEY = "YQNUKAH419JA2RYV"

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

st.set_page_config(layout="wide")

# =========================
# GET KURS REAL-TIME
# =========================
@st.cache_data(ttl=3600)
def get_kurs_usd_idr():
    try:
        url = "https://api.exchangerate.host/latest?base=USD&symbols=IDR"
        res = requests.get(url).json()
        return res['rates']['IDR']
    except:
        return 15500  # fallback

# =========================
# PREPROCESS
# =========================
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return stopword.remove(text)

# =========================
# NORMALIZE DATA
# =========================
def normalize_stock_df(df):
    if df is None or df.empty:
        return None

    df = df.reset_index()

    if 'Date' not in df.columns:
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    return df

# =========================
# YAHOO
# =========================
@st.cache_data(ttl=300)
def get_yahoo(ticker, start, end):
    try:
        ticker_jk = ticker if ticker.endswith(".JK") else ticker + ".JK"
        time.sleep(1)
        df = yf.Ticker(ticker_jk).history(start=start, end=end)
        if df.empty:
            return None
        return normalize_stock_df(df)
    except:
        return None

# =========================
# ALPHA
# =========================
@st.cache_data(ttl=300)
def get_alpha(ticker, start, end):
    try:
        ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=ticker, outputsize='compact')

        df = data.rename(columns={'4. close': 'Close'})
        df.index = pd.to_datetime(df.index)

        df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]

        if df.empty:
            return None

        return normalize_stock_df(df)

    except:
        return None

# =========================
# MULTI SOURCE
# =========================
def get_stock_data(source, ticker, start, end):

    if source == "Yahoo Finance":
        df = get_yahoo(ticker, start, end)
        if df is None:
            st.warning("⚠️ Yahoo gagal → fallback Alpha")
            df = get_alpha(ticker, start, end)
    else:
        df = get_alpha(ticker, start, end)
        if df is None:
            st.warning("⚠️ Alpha gagal → fallback Yahoo")
            df = get_yahoo(ticker, start, end)

    if df is None:
        return None

    kurs = get_kurs_usd_idr()

    df['Prev_Close'] = df['Close'].shift(1)
    df['Price_Change'] = df['Close'] - df['Prev_Close']
    df['Pct_Change (%)'] = (df['Price_Change'] / df['Prev_Close']) * 100

    # KONVERSI RUPIAH
    df['Close_IDR'] = df['Close'] * kurs

    return df

# =========================
# STYLE WARNA
# =========================
def highlight_change(val):
    if pd.isna(val):
        return ""
    if val > 0:
        return "color: green"
    elif val < 0:
        return "color: red"
    return ""

# =========================
# NEWS
# =========================
@st.cache_data(ttl=300)
def crawl_news(keyword):
    url = f"https://news.google.com/rss/search?q={keyword.replace(' ', '%20')}&hl=id&gl=ID&ceid=ID:id"
    feed = feedparser.parse(url)

    if not feed.entries:
        return None, None, None, None

    data = []
    for e in feed.entries:
        data.append({
            "title": e.title,
            "link": e.link,
            "published": e.get("published", ""),
            "source": e.get("source", {}).get("title", "")
        })

    df = pd.DataFrame(data)
    df["doc"] = df["title"].apply(preprocess_text)

    text = " ".join(df["doc"])

    wc, common = None, None
    if text.strip():
        wc = WordCloud(background_color="white").generate(text)
        common = Counter(text.split()).most_common(10)

    media = df["source"].value_counts().head(10)

    return df, wc, common, media

# =========================
# UI
# =========================
st.title("💹 Dashboard Saham & Berita (PRO MAX)")

st.sidebar.header("⚙️ Pengaturan")

source = st.sidebar.selectbox("Sumber Data", ["Yahoo Finance", "Alpha Vantage"])
ticker = st.sidebar.text_input("Ticker", "BBCA")
keyword = st.sidebar.text_input("Keyword Berita", "Bank BCA")

c1, c2 = st.sidebar.columns(2)
start = c1.date_input("Mulai", datetime.now() - timedelta(days=30))
end = c2.date_input("Selesai", datetime.now())

if st.sidebar.button("🚀 Jalankan"):

    df_s = get_stock_data(source, ticker, start, end)
    df_n, wc, common, media = crawl_news(keyword)

    tab1, tab2 = st.tabs(["📉 Saham", "📰 Berita"])

    # ================= SAHAM =================
    with tab1:
        if df_s is not None:

            st.success(f"✅ Data saham: {len(df_s)}")

            cols = ['Date','Prev_Close','Close','Close_IDR','Price_Change','Pct_Change (%)']
            cols = [c for c in cols if c in df_s.columns]

            styled = df_s[cols].style.format({
                "Prev_Close": "{:.2f}",
                "Close": "{:.2f}",
                "Close_IDR": "Rp {:,.0f}",
                "Price_Change": "{:.2f}",
                "Pct_Change (%)": "{:.2f}"
            }).applymap(highlight_change, subset=['Price_Change','Pct_Change (%)'])

            st.dataframe(styled, use_container_width=True)

            # GRAFIK RUPIAH
            st.subheader("📈 Grafik Harga (Rupiah)")
            st.line_chart(df_s.set_index("Date")["Close_IDR"])

        else:
            st.error("❌ Semua source gagal")

    # ================= BERITA =================
    with tab2:
        if df_n is not None:

            st.success(f"✅ Berita ditemukan: {len(df_n)}")
            st.dataframe(df_n, use_container_width=True)

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
            st.warning("⚠️ Berita tidak ditemukan")
