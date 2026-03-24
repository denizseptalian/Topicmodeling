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

# =========================
# CONFIG
# =========================
AV_API_KEY = "YQNUKAH419JA2RYV"

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

st.set_page_config(layout="wide")

# =========================
# PREPROCESS
# =========================
def preprocess_text(text):
    try:
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return stopword.remove(text)
    except:
        return ""

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

        df = df.reset_index()
        return df

    except:
        return None

# =========================
# ALPHA VANTAGE
# =========================
@st.cache_data(ttl=300)
def get_alpha(ticker, start, end):
    try:
        ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=ticker, outputsize='compact')

        df = data.rename(columns={'4. close': 'Close'})
        df.index = pd.to_datetime(df.index)

        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

        df = df.loc[(df.index >= start) & (df.index <= end)]
        df = df.reset_index()

        return df

    except:
        return None

# =========================
# SMART SOURCE
# =========================
def get_stock_data(source, ticker, start, end):

    if source == "Yahoo Finance":
        df = get_yahoo(ticker, start, end)
        if df is None:
            st.warning("⚠️ Yahoo gagal → fallback ke Alpha Vantage")
            df = get_alpha(ticker, start, end)
    else:
        df = get_alpha(ticker, start, end)
        if df is None:
            st.warning("⚠️ Alpha gagal → coba Yahoo")
            df = get_yahoo(ticker, start, end)

    if df is None:
        return None

    df['Prev_Close'] = df['Close'].shift(1)
    df['Price_Change'] = df['Close'] - df['Prev_Close']
    df['Pct_Change (%)'] = (df['Price_Change'] / df['Prev_Close']) * 100

    return df

# =========================
# NEWS FULL FEATURE
# =========================
@st.cache_data(ttl=300)
def crawl_news(keyword):
    try:
        url = f"https://news.google.com/rss/search?q={keyword.replace(' ', '%20')}&hl=id&gl=ID&ceid=ID:id"
        feed = feedparser.parse(url)

        if not feed.entries:
            return None, None, None, None

        data = []

        for entry in feed.entries:
            data.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.get("published", ""),
                "source": entry.get("source", {}).get("title", "")
            })

        df = pd.DataFrame(data)

        # TEXT PROCESSING
        df["document"] = df["title"].apply(preprocess_text)

        text = " ".join(df["document"])

        wordcloud, common = None, None

        if text.strip():
            wordcloud = WordCloud(
                background_color="white",
                max_words=5000
            ).generate(text)

            common = Counter(text.split()).most_common(10)

        # TOP MEDIA
        top_media = df["source"].value_counts().head(10)

        return df, wordcloud, common, top_media

    except Exception as e:
        st.error(f"Error news: {e}")
        return None, None, None, None

# =========================
# UI
# =========================
st.title("💹 Dashboard Saham & Berita (MULTI SOURCE PRO)")

st.sidebar.header("⚙️ Pengaturan")

source = st.sidebar.selectbox(
    "Sumber Data Saham",
    ["Yahoo Finance", "Alpha Vantage"]
)

ticker = st.sidebar.text_input("Ticker Saham", "BBCA")
keyword = st.sidebar.text_input("Keyword Berita", "Bank BCA")

c1, c2 = st.sidebar.columns(2)
start = c1.date_input("Mulai", datetime.now() - timedelta(days=30))
end = c2.date_input("Selesai", datetime.now())

if st.sidebar.button("🚀 Jalankan Analisis"):

    with st.spinner("Mengambil data..."):

        df_s = get_stock_data(source, ticker, start, end)
        df_n, wc, common, media = crawl_news(keyword)

        tab1, tab2 = st.tabs(["📉 Saham", "📰 Berita"])

        # =========================
        # SAHAM
        # =========================
        with tab1:
            if df_s is not None:

                st.success(f"✅ Data saham: {len(df_s)} hari")

                st.dataframe(
                    df_s[['Date','Close','Prev_Close','Price_Change','Pct_Change (%)']]
                    .style.format("{:.2f}"),
                    use_container_width=True
                )

                st.subheader("📈 Grafik Close")
                st.line_chart(df_s.set_index("Date")['Close'])

            else:
                st.error("❌ Semua sumber gagal")

        # =========================
        # BERITA (FULL FEATURE)
        # =========================
        with tab2:
            if df_n is not None:

                st.success(f"✅ Berita ditemukan: {len(df_n)}")

                st.dataframe(df_n, use_container_width=True)

                if wc:
                    st.subheader("☁️ WordCloud")
                    fig, ax = plt.subplots(figsize=(10,5))
                    ax.imshow(wc)
                    ax.axis("off")
                    st.pyplot(fig)

                if common:
                    st.subheader("📌 10 Kata Teratas")
                    st.table(pd.DataFrame(common, columns=["Kata","Jumlah"]))

                if not media.empty:
                    st.subheader("🏆 Top Media")
                    st.table(
                        media.reset_index()
                        .rename(columns={"index":"Media", "source":"Jumlah"})
                    )

            else:
                st.warning("⚠️ Berita tidak ditemukan")
