import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from wordcloud import WordCloud
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
from datetime import datetime, timedelta
import feedparser

# =========================
# STOPWORD
# =========================
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
# YAHOO FINANCE (STABLE)
# =========================
def get_stock_data_yf(ticker, start_date, end_date):
    try:
        # auto tambah .JK
        if not ticker.endswith(".JK"):
            ticker = ticker + ".JK"

        # retry 3x (ANTI ERROR YFINANCE)
        for i in range(3):
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                threads=False
            )

            if df is not None and not df.empty:
                break

        if df is None or df.empty:
            st.error("❌ Data tidak ditemukan di Yahoo Finance")
            return None

        # FIX timezone issue
        df = df.reset_index()

        df['Prev_Close'] = df['Close'].shift(1)
        df['Price_Change'] = df['Close'] - df['Prev_Close']
        df['Pct_Change (%)'] = (df['Price_Change'] / df['Prev_Close']) * 100

        return df

    except Exception as e:
        st.error(f"Error Yahoo Finance: {e}")
        return None

# =========================
# NEWS RSS (STABLE)
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

        top_media = df["source"].value_counts().head(10)

        return df, wordcloud, common, top_media

    except Exception as e:
        st.error(f"Error news: {e}")
        return None, None, None, None

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(layout="wide")
st.title("💹 Dashboard Saham & Berita (Yahoo Finance + RSS)")

# Sidebar
ticker = st.sidebar.text_input("Ticker Saham", "BBCA")
keyword = st.sidebar.text_input("Keyword Berita", "Bank BCA")

c1, c2 = st.sidebar.columns(2)
start_d = c1.date_input("Mulai", datetime.now() - timedelta(days=30))
end_d = c2.date_input("Selesai", datetime.now())

if st.sidebar.button("🚀 Jalankan"):

    with st.spinner("Ambil data..."):

        df_s = get_stock_data_yf(ticker, start_d, end_d)
        df_n, wc, common, media = crawl_news_rss(keyword)

        tab1, tab2 = st.tabs(["📉 Saham", "📰 Berita"])

        # =========================
        # SAHAM
        # =========================
        with tab1:
            if df_s is not None:

                st.success(f"✅ Data saham: {len(df_s)} hari")

                st.dataframe(
                    df_s[['Close', 'Prev_Close', 'Price_Change', 'Pct_Change (%)']]
                    .style.format("{:.2f}"),
                    use_container_width=True
                )

                st.subheader("📈 Grafik Close")
                st.line_chart(df_s['Close'])

            else:
                st.warning("⚠️ Data saham tidak ditemukan")

        # =========================
        # BERITA
        # =========================
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
