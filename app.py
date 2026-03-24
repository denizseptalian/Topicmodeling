import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
import streamlit as st
from wordcloud import WordCloud
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
from datetime import datetime, timedelta
import feedparser

# =========================
# CONFIG
# =========================
AV_API_KEY = "CYJG0OMG7PWSU1V9"

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# =========================
# PREPROCESSING
# =========================
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return stopword.remove(text)

# =========================
# STOCK DATA
# =========================
def get_stock_data_av(ticker, start_date, end_date):
    try:
        ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=ticker, outputsize='compact')

        df = data.rename(columns={'4. close': 'Close'})
        df.index = pd.to_datetime(df.index)

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        df = df.loc[(df.index >= start_date) & (df.index <= end_date)]

        df['Prev_Close'] = df['Close'].shift(1)
        df['Price_Change'] = df['Close'] - df['Prev_Close']
        df['Pct_Change (%)'] = (df['Price_Change'] / df['Prev_Close']) * 100

        return df

    except Exception as e:
        st.error(f"Error saham: {e}")
        return None

# =========================
# NEWS (RSS VERSION - STABLE)
# =========================
def crawl_news_rss(keyword):
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

        df["document"] = df["title"].apply(preprocess_text)

        text = " ".join(df["document"])

        wordcloud, common = None, None

        if text.strip():
            wordcloud = WordCloud(background_color="white").generate(text)
            common = Counter(text.split()).most_common(10)

        # TOP MEDIA
        top_media = df["source"].value_counts().head(10)

        return df, wordcloud, common, top_media

    except Exception as e:
        st.error(f"Error news RSS: {e}")
        return None, None, None, None

# =========================
# UI
# =========================
st.set_page_config(layout="wide")
st.title("💹 Dashboard Saham & Berita (RSS FIX)")

ticker = st.sidebar.text_input("Ticker", "BBCA")
keyword = st.sidebar.text_input("Keyword", "Bank BCA")

c1, c2 = st.sidebar.columns(2)
start_d = c1.date_input("Mulai", datetime.now() - timedelta(days=20))
end_d = c2.date_input("Selesai", datetime.now())

if st.sidebar.button("Jalankan"):

    df_s = get_stock_data_av(ticker, start_d, end_d)
    df_n, wc, common, media = crawl_news_rss(keyword)

    tab1, tab2 = st.tabs(["Saham", "Berita"])

    # SAHAM
    with tab1:
        if df_s is not None:
            st.dataframe(df_s)
            st.line_chart(df_s["Close"])
        else:
            st.warning("Data saham kosong")

    # BERITA
    with tab2:
        if df_n is not None:
            st.success(f"✅ Berita ditemukan: {len(df_n)}")
            st.dataframe(df_n)

            if wc:
                fig, ax = plt.subplots()
                ax.imshow(wc)
                ax.axis("off")
                st.pyplot(fig)

            if common:
                st.table(pd.DataFrame(common, columns=["Kata","Jumlah"]))

            if not media.empty:
                st.table(media.reset_index().rename(columns={"index":"Media", "source":"Jumlah"}))
        else:
            st.error("❌ RSS juga tidak menemukan berita")
