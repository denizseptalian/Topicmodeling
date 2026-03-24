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
import requests

# =========================
# CONFIG & INITIALIZATION
# =========================
st.set_page_config(layout="wide", page_title="Analisis Sentimen & Saham")

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

def preprocess_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return stopword.remove(text)

# =========================
# STOCK DATA (YAHOO FIX)
# =========================
def get_stock_data_yahoo(ticker, start_date, end_date):
    try:
        # Tambah buffer hari agar Prev_Close tidak kosong di hari pertama
        start_dt = pd.to_datetime(start_date) - timedelta(days=14)
        end_dt = pd.to_datetime(end_date) + timedelta(days=1)
        
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})

        df = yf.download(
            ticker, 
            start=start_dt.strftime('%Y-%m-%d'), 
            end=end_dt.strftime('%Y-%m-%d'), 
            session=session,
            progress=False,
            auto_adjust=True
        )

        if df.empty: return None

        # Handle MultiIndex Columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.sort_index()
        df['Prev_Close'] = df['Close'].shift(1)
        df['Price_Change'] = df['Close'] - df['Prev_Close']
        df['Pct_Change (%)'] = (df['Price_Change'] / df['Prev_Close']) * 100

        # Kembalikan ke range tanggal yang dipilih user
        df.index = pd.to_datetime(df.index)
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        return df.loc[mask]

    except Exception as e:
        st.error(f"Error Saham: {e}")
        return None

# =========================
# NEWS (RSS VERSION)
# =========================
def crawl_news_rss(keyword):
    try:
        url = f"https://news.google.com/rss/search?q={keyword.replace(' ', '%20')}&hl=id&gl=ID&ceid=ID:id"
        feed = feedparser.parse(url)
        if not feed.entries: return None, None, None, None

        data = []
        for entry in feed.entries:
            data.append({
                "Tanggal": entry.get("published", ""),
                "Judul": entry.title,
                "Sumber": entry.get("source", {}).get("title", ""),
                "Link": entry.link
            })

        df = pd.DataFrame(data)
        df["doc"] = df["Judul"].apply(preprocess_text)
        txt = " ".join(df["doc"])
        
        wc = WordCloud(background_color="white", width=800, height=400).generate(txt) if txt.strip() else None
        common = Counter(txt.split()).most_common(10)
        top_media = df["Sumber"].value_counts().head(10)

        return df, wc, common, top_media
    except:
        return None, None, None, None

# =========================
# UI
# =========================
st.title("💹 Analisis Sentimen Berita & Pergerakan Saham")

# Sidebar
st.sidebar.header("Konfigurasi")
ticker = st.sidebar.text_input("Ticker (Contoh: BBCA.JK atau ASHA.JK):", value="BBCA.JK")
keyword = st.sidebar.text_input("Kata Kunci Berita:", value="Bank BCA")

c1, c2 = st.sidebar.columns(2)
start_d = c1.date_input("Mulai", datetime.now() - timedelta(days=14))
end_d = c2.date_input("Selesai", datetime.now())

if st.sidebar.button("🚀 Jalankan Analisis"):
    df_s = get_stock_data_yahoo(ticker, start_d, end_d)
    df_n, wc, common, media = crawl_news_rss(keyword)

    tab1, tab2, tab3 = st.tabs(["📉 Pergerakan Saham", "📰 Daftar Berita", "📊 Analisis Kata & Sentimen"])

    with tab1:
        if df_s is not None and not df_s.empty:
            st.subheader(f"Data Harga {ticker}")
            st.dataframe(df_s[['Prev_Close', 'Close', 'Price_Change', 'Pct_Change (%)']].style.format("{:.2f}"), use_container_width=True)
            st.line_chart(df_s["Close"])
        else:
            st.error("Gagal menarik data saham. Pastikan ticker benar (Gunakan .JK) dan bursa sedang buka.")

    with tab2:
        if df_n is not None:
            st.success(f"Ditemukan {len(df_n)} berita.")
            st.dataframe(df_n[["Tanggal", "Judul", "Sumber", "Link"]], use_container_width=True)
        else:
            st.warning("Berita tidak ditemukan.")

    with tab3:
        if wc:
            st.subheader("WordCloud (Kata Kunci Berita)")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("10 Kata Terbanyak")
                st.table(pd.DataFrame(common, columns=["Kata", "Jumlah"]))
            with col_b:
                st.subheader("Top Media")
                st.bar_chart(media)
