import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from GoogleNews import GoogleNews
import streamlit as st
from wordcloud import WordCloud
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
from datetime import datetime, timedelta
import logging

# --- KONFIGURASI ---
AV_API_KEY = "CYJG0OMG7PWSU1V9"

# Stopword
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# =========================
# PREPROCESSING
# =========================
def preprocess_text(text):
    if pd.isna(text) or text == "":
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return stopword.remove(text)

# =========================
# SAHAM (ALPHA VANTAGE)
# =========================
def get_stock_data_av(ticker, start_date, end_date):
    try:
        ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=ticker, outputsize='compact')

        df = data.rename(columns={'4. close': 'Close'})
        df = df.sort_index()

        df['Prev_Close'] = df['Close'].shift(1)
        df['Price_Change'] = df['Close'] - df['Prev_Close']
        df['Pct_Change (%)'] = (df['Price_Change'] / df['Prev_Close']) * 100

        df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
        df_filtered = df.loc[(df.index >= start_date) & (df.index <= end_date)]

        return df_filtered

    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        return None

# =========================
# NEWS + ANALYSIS (BARU)
# =========================
def crawl_and_analyze(keyword, start_date, end_date):

    googlenews = GoogleNews(lang='id', region='ID')
    search_q = f"{keyword} after:{start_date} before:{end_date}"
    googlenews.search(search_q)

    data_to_append = []

    for i in range(1, 6):
        try:
            googlenews.getpage(i)
            news = googlenews.results()

            if len(news) == 0:
                continue

            df_temp = pd.DataFrame(news)
            df_temp = df_temp.drop_duplicates(subset="title")

            data_to_append.append(df_temp)

        except Exception as e:
            logging.warning(f"Gagal page {i}: {e}")

    if len(data_to_append) == 0:
        return None, None, None, None

    df = pd.concat(data_to_append, ignore_index=True)

    df["title"] = df["title"].fillna("") if "title" in df else ""
    df["desc"] = df["desc"].fillna("") if "desc" in df else ""

    df["document"] = (df["title"] + " " + df["desc"]).apply(preprocess_text)

    long_string = " ".join(df["document"].values)

    wordcloud = None
    most_common_words = None

    if long_string.strip() != "":
        wordcloud = WordCloud(background_color="white").generate(long_string)
        words = long_string.split()
        most_common_words = Counter(words).most_common(10)

    # MEDIA ANALYSIS
    if "media" in df.columns:
        top_media = df["media"].value_counts().head(10)
    elif "site" in df.columns:
        top_media = df["site"].value_counts().head(10)
    else:
        top_media = pd.Series()

    return df, wordcloud, most_common_words, top_media

# =========================
# UI STREAMLIT
# =========================
st.set_page_config(page_title="Dashboard Saham & Berita", layout="wide")

st.title("💹 Dashboard Saham + Analisis Berita")

# Sidebar
st.sidebar.header("⚙️ Pengaturan")

ticker = st.sidebar.text_input("Ticker Saham", "BBCA")
keyword = st.sidebar.text_input("Keyword Berita", "Bank Central Asia")

c1, c2 = st.sidebar.columns(2)
start_d = c1.date_input("Mulai", datetime.now() - timedelta(days=20))
end_d = c2.date_input("Selesai", datetime.now())

if st.sidebar.button("🚀 Jalankan"):

    s_str = start_d.strftime("%Y-%m-%d")
    e_str = end_d.strftime("%Y-%m-%d")

    with st.spinner("Processing..."):

        df_s = get_stock_data_av(ticker, s_str, e_str)
        df_n, wc, common, media = crawl_and_analyze(keyword, s_str, e_str)

        tab1, tab2 = st.tabs(["📉 Saham", "📰 Berita"])

        # =========================
        # TAB SAHAM
        # =========================
        with tab1:
            if df_s is not None and not df_s.empty:

                st.subheader("Data Harga")
                st.dataframe(df_s)

                st.subheader("Grafik Close")
                st.line_chart(df_s["Close"])

            else:
                st.error("Data saham tidak ditemukan")

        # =========================
        # TAB BERITA
        # =========================
        with tab2:
            if df_n is not None:

                st.subheader("Data Berita")
                st.dataframe(df_n[['date','title','media','link']], use_container_width=True)

                # WORDCLOUD
                if wc:
                    st.subheader("WordCloud")
                    fig, ax = plt.subplots()
                    ax.imshow(wc)
                    ax.axis("off")
                    st.pyplot(fig)

                # MOST COMMON WORDS
                if common:
                    st.subheader("Top 10 Kata")
                    st.table(pd.DataFrame(common, columns=["Kata","Jumlah"]))

                # TOP MEDIA
                if not media.empty:
                    st.subheader("Top Media")
                    st.table(media.reset_index().rename(columns={"index":"Media", 0:"Jumlah"}))

            else:
                st.warning("Tidak ada berita ditemukan")
