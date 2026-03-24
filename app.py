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

# =========================
# CONFIG
# =========================
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
# STOCK DATA
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

        df.index = pd.to_datetime(df.index)
        df = df.loc[(df.index >= start_date) & (df.index <= end_date)]

        return df

    except Exception as e:
        st.error(f"Error saham: {e}")
        return None

# =========================
# NEWS CRAWLING (FIXED)
# =========================
def crawl_and_analyze(keyword):

    googlenews = GoogleNews(lang='id')  # lebih stabil tanpa region
    googlenews.search(keyword)

    data_to_append = []

    for i in range(1, 4):  # max 3 page biar stabil
        try:
            googlenews.getpage(i)
            news = googlenews.results()

            if not news:
                continue

            df_temp = pd.DataFrame(news)
            df_temp = df_temp.drop_duplicates(subset="title")

            data_to_append.append(df_temp)

        except Exception as e:
            logging.warning(f"Gagal page {i}: {e}")

    if not data_to_append:
        return None, None, None, None

    df = pd.concat(data_to_append, ignore_index=True)

    # Safe column handling
    df["title"] = df.get("title", "").fillna("")
    df["desc"] = df.get("desc", "").fillna("")

    # TEXT PROCESS
    df["document"] = (df["title"] + " " + df["desc"]).apply(preprocess_text)

    long_string = " ".join(df["document"].values)

    wordcloud, most_common_words = None, None

    if long_string.strip():
        wordcloud = WordCloud(background_color="white").generate(long_string)
        most_common_words = Counter(long_string.split()).most_common(10)

    # MEDIA ANALYSIS
    media_col = "media" if "media" in df.columns else "site" if "site" in df.columns else None
    top_media = df[media_col].value_counts().head(10) if media_col else pd.Series()

    return df, wordcloud, most_common_words, top_media

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Dashboard Saham & Berita", layout="wide")

st.title("💹 Dashboard Analisis Saham & Berita")

# Sidebar
st.sidebar.header("⚙️ Pengaturan")

ticker = st.sidebar.text_input("Ticker Saham", "BBCA")
keyword = st.sidebar.text_input("Keyword Berita", "Bank BCA")

c1, c2 = st.sidebar.columns(2)
start_d = c1.date_input("Mulai", datetime.now() - timedelta(days=20))
end_d = c2.date_input("Selesai", datetime.now())

if st.sidebar.button("🚀 Jalankan Analisis"):

    with st.spinner("Mengambil data..."):

        # STOCK
        df_s = get_stock_data_av(ticker, start_d, end_d)

        # NEWS
        df_n, wc, common, media = crawl_and_analyze(keyword)

        tab1, tab2 = st.tabs(["📉 Saham", "📰 Berita"])

        # =========================
        # TAB SAHAM
        # =========================
        with tab1:
            if df_s is not None and not df_s.empty:

                st.subheader("Data Harga Saham")
                st.dataframe(
                    df_s[['Prev_Close', 'Close', 'Price_Change', 'Pct_Change (%)']]
                    .style.format("{:.2f}"),
                    use_container_width=True
                )

                st.subheader("Grafik Harga Close")
                st.line_chart(df_s['Close'])

            else:
                st.error("❌ Data saham tidak ditemukan")

        # =========================
        # TAB BERITA
        # =========================
        with tab2:
            if df_n is not None:

                st.subheader("Data Berita")
                st.write(f"Jumlah berita ditemukan: {len(df_n)}")

                cols = [c for c in ['date', 'title', 'media', 'link'] if c in df_n.columns]
                st.dataframe(df_n[cols], use_container_width=True)

                # WORDCLOUD
                if wc:
                    st.subheader("☁️ WordCloud")
                    fig, ax = plt.subplots(figsize=(10,5))
                    ax.imshow(wc)
                    ax.axis("off")
                    st.pyplot(fig)

                # TOP WORDS
                if common:
                    st.subheader("📌 10 Kata Teratas")
                    st.table(pd.DataFrame(common, columns=["Kata", "Jumlah"]))

                # TOP MEDIA
                if not media.empty:
                    st.subheader("🏆 Media Teraktif")
                    st.table(
                        media.reset_index()
                        .rename(columns={"index":"Media", 0:"Jumlah"})
                    )

            else:
                st.warning("⚠️ Berita tidak ditemukan (coba keyword lain)")
