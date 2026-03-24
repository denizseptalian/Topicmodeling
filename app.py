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

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# =========================
# PREPROCESSING
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
# STOCK DATA
# =========================
def get_stock_data_av(ticker, start_date, end_date):
    try:
        ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=ticker, outputsize='compact')

        if data is None or data.empty:
            return None

        df = data.rename(columns={'4. close': 'Close'})
        df = df.sort_index()

        df.index = pd.to_datetime(df.index, errors='coerce')

        # FIX datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        df = df.loc[(df.index >= start_date) & (df.index <= end_date)]

        if df.empty:
            return None

        df['Prev_Close'] = df['Close'].shift(1)
        df['Price_Change'] = df['Close'] - df['Prev_Close']
        df['Pct_Change (%)'] = (df['Price_Change'] / df['Prev_Close']) * 100

        return df

    except Exception as e:
        st.error(f"Error saham: {e}")
        return None

# =========================
# NEWS (SUPER FIX)
# =========================
def crawl_and_analyze(keyword):
    try:
        # FIX keyword biar lebih kuat
        keyword = keyword.title()

        googlenews = GoogleNews(lang='id', period='7d')  # ambil 7 hari terakhir
        googlenews.clear()
        googlenews.search(keyword)

        all_news = []

        for i in range(1, 6):  # lebih banyak page
            try:
                googlenews.getpage(i)
                news = googlenews.results()

                if news:
                    all_news.extend(news)

            except Exception as e:
                logging.warning(f"Error page {i}: {e}")

        if not all_news:
            st.error("❌ DEBUG: GoogleNews tidak mengembalikan data")
            return None, None, None, None

        df = pd.DataFrame(all_news)

        # DEBUG
        st.write("📊 DEBUG jumlah berita:", len(df))

        df = df.drop_duplicates(subset="title", errors='ignore')

        df["title"] = df.get("title", "").fillna("")
        df["desc"] = df.get("desc", "").fillna("")

        df["document"] = (df["title"] + " " + df["desc"]).apply(preprocess_text)

        text = " ".join(df["document"].values)

        wordcloud, common = None, None

        if text.strip():
            wordcloud = WordCloud(background_color="white").generate(text)
            common = Counter(text.split()).most_common(10)

        # MEDIA
        media_col = "media" if "media" in df.columns else "site" if "site" in df.columns else None
        top_media = df[media_col].value_counts().head(10) if media_col else pd.Series()

        return df, wordcloud, common, top_media

    except Exception as e:
        st.error(f"Error news: {e}")
        return None, None, None, None

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Dashboard Saham & Berita", layout="wide")

st.title("💹 Dashboard Analisis Saham & Berita (FINAL FIX)")

# Sidebar
st.sidebar.header("⚙️ Pengaturan")

ticker = st.sidebar.text_input("Ticker Saham", "BBCA")
keyword = st.sidebar.text_input("Keyword Berita", "Bank BCA")

c1, c2 = st.sidebar.columns(2)
start_d = c1.date_input("Mulai", datetime.now() - timedelta(days=20))
end_d = c2.date_input("Selesai", datetime.now())

if st.sidebar.button("🚀 Jalankan Analisis"):

    with st.spinner("Processing..."):

        # STOCK
        df_s = get_stock_data_av(ticker, start_d, end_d)

        # NEWS
        df_n, wc, common, media = crawl_and_analyze(keyword)

        tab1, tab2 = st.tabs(["📉 Saham", "📰 Berita"])

        # =========================
        # SAHAM
        # =========================
        with tab1:
            if df_s is not None and not df_s.empty:
                st.success(f"✅ Data saham: {len(df_s)} baris")

                st.dataframe(
                    df_s[['Prev_Close', 'Close', 'Price_Change', 'Pct_Change (%)']]
                    .style.format("{:.2f}"),
                    use_container_width=True
                )

                st.subheader("📈 Grafik Close")
                st.line_chart(df_s['Close'])
            else:
                st.warning("⚠️ Data saham kosong")

        # =========================
        # BERITA
        # =========================
        with tab2:
            if df_n is not None and not df_n.empty:

                st.success(f"✅ Berita ditemukan: {len(df_n)}")

                cols = [c for c in ['date','title','media','link'] if c in df_n.columns]
                st.dataframe(df_n[cols], use_container_width=True)

                if wc:
                    st.subheader("☁️ WordCloud")
                    fig, ax = plt.subplots(figsize=(10,5))
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
                        .rename(columns={"index":"Media", 0:"Jumlah"})
                    )

            else:
                st.warning("⚠️ Berita tidak ditemukan")
