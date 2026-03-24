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
st.set_page_config(layout="wide", page_title="Dashboard Saham & Berita")

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
# STOCK DATA (YAHOO FINANCE FIX)
# =========================
def get_stock_data_yahoo(ticker, start_date, end_date):
    try:
        # Mundurkan tanggal sedikit untuk kalkulasi Prev Close
        start_dt = pd.to_datetime(start_date) - timedelta(days=10)
        
        # Gunakan Session agar tidak diblokir server
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

        df = yf.download(
            ticker, 
            start=start_dt.strftime('%Y-%m-%d'), 
            end=end_date, 
            session=session,
            progress=False,
            auto_adjust=True
        )

        if df.empty:
            return None

        # Perbaikan struktur kolom jika MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.sort_index()
        
        # Kalkulasi kolom tambahan
        df['Prev_Close'] = df['Close'].shift(1)
        df['Price_Change'] = df['Close'] - df['Prev_Close']
        df['Pct_Change (%)'] = (df['Price_Change'] / df['Prev_Close']) * 100

        # Filter kembali ke tanggal yang diminta user
        df = df.loc[df.index >= pd.to_datetime(start_date)]
        
        return df

    except Exception as e:
        st.error(f"Error penarikan data saham: {e}")
        return None

# =========================
# NEWS (RSS VERSION - STABLE)
# =========================
def crawl_news_rss(keyword):
    try:
        # Google News RSS bersifat publik dan tidak ada API limit yang ketat
        url = f"https://news.google.com/rss/search?q={keyword.replace(' ', '%20')}&hl=id&gl=ID&ceid=ID:id"
        feed = feedparser.parse(url)

        if not feed.entries:
            return None, None, None, None

        data = []
        for entry in feed.entries:
            data.append({
                "Tanggal": entry.get("published", ""),
                "Judul": entry.title,
                "Sumber": entry.get("source", {}).get("title", ""),
                "Link": entry.link
            })

        df = pd.DataFrame(data)
        
        # Preprocessing untuk WordCloud
        df["document"] = df["Judul"].apply(preprocess_text)
        full_text = " ".join(df["document"])

        wordcloud, common = None, None
        if full_text.strip():
            wordcloud = WordCloud(background_color="white", width=800, height=400).generate(full_text)
            common = Counter(full_text.split()).most_common(10)

        top_media = df["Sumber"].value_counts().head(10)

        return df, wordcloud, common, top_media

    except Exception as e:
        st.error(f"Error news RSS: {e}")
        return None, None, None, None

# =========================
# USER INTERFACE
# =========================
st.title("💹 Analisis Sentimen Berita & Pergerakan Saham")
st.caption("Data Saham: Yahoo Finance | Berita: Google News RSS")

# SIDEBAR
st.sidebar.header("Pengaturan Analisis")
ticker = st.sidebar.text_input("Simbol Saham (Contoh: BBCA.JK atau ASHA.JK):", value="BBCA.JK")
keyword = st.sidebar.text_input("Kata Kunci Berita:", value="Bank BCA")

col_date1, col_date2 = st.sidebar.columns(2)
start_d = col_date1.date_input("Mulai", datetime.now() - timedelta(days=20))
end_d = col_date2.date_input("Selesai", datetime.now())

if st.sidebar.button("🚀 Jalankan Analisis"):
    
    # 1. Ambil Data Saham
    df_s = get_stock_data_yahoo(ticker, start_d, end_d)
    
    # 2. Ambil Data Berita
    df_n, wc, common, media = crawl_news_rss(keyword)

    tab1, tab2, tab3 = st.tabs(["📉 Pergerakan Saham", "📰 Daftar Berita", "📊 Analisis Kata"])

    # --- TAB 1: SAHAM ---
    with tab1:
        if df_s is not None and not df_s.empty:
            st.subheader(f"Tabel Harga {ticker}")
            st.dataframe(df_s[['Prev_Close', 'Close', 'Price_Change', 'Pct_Change (%)']].style.format("{:.2f}"), use_container_width=True)
            
            st.subheader(f"Grafik Harga Penutupan (Close) - {ticker}")
            st.line_chart(df_s["Close"])
        else:
            st.warning("Data saham tidak ditemukan. Pastikan ticker menggunakan akhiran .JK untuk bursa Indonesia.")

    # --- TAB 2: BERITA ---
    with tab2:
        if df_n is not None:
            st.success(f"Ditemukan {len(df_n)} berita terbaru.")
            st.dataframe(df_n[["Tanggal", "Judul", "Sumber", "Link"]], use_container_width=True)
        else:
            st.error("Tidak ada berita yang ditemukan untuk kata kunci tersebut.")

    # --- TAB 3: ANALISIS KATA ---
    with tab3:
        col_wc, col_freq = st.columns(2)
        
        if wc:
            with col_wc:
                st.subheader("WordCloud")
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
        
        if common:
            with col_freq:
                st.subheader("10 Kata Terbanyak")
                df_common = pd.DataFrame(common, columns=["Kata", "Jumlah"])
                st.table(df_common)
        
        if media is not None and not media.empty:
            st.subheader("Top Media Pemberitaan")
            st.bar_chart(media)
