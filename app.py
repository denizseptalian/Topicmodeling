import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from GoogleNews import GoogleNews
import streamlit as st
from wordcloud import WordCloud
import logging
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
import requests
from datetime import datetime, timedelta

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO)

# Inisialisasi Sastrawi
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

def preprocess_text(text):
    if pd.isna(text): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = stopword.remove(text)
    return text

def get_stock_data(ticker, start_date, end_date):
    try:
        # Ambil data H-7 untuk mendapatkan Prev Close yang akurat
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=7)
        
        # Penyamaran User-Agent untuk menembus proteksi Yahoo Finance di Cloud
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Download data menggunakan session khusus
        df_stock = yf.download(
            ticker, 
            start=start_dt.strftime("%Y-%m-%d"), 
            end=end_date, 
            auto_adjust=True,
            session=session
        )
        
        if df_stock.empty or len(df_stock) < 2:
            st.sidebar.warning(f"Data {ticker} tidak ditemukan atau terlalu sedikit.")
            return None

        # Fix MultiIndex Columns (Sangat Penting untuk yfinance v0.2+)
        if isinstance(df_stock.columns, pd.MultiIndex):
            df_stock.columns = df_stock.columns.get_level_values(0)

        df_final = df_stock.copy()
        
        if 'Close' in df_final.columns:
            # Perhitungan harga sesuai permintaan Anda
            df_final['Prev_Close'] = df_final['Close'].shift(1)
            df_final['Price_Change'] = df_final['Close'] - df_final['Prev_Close']
            df_final['Pct_Change (%)'] = (df_final['Price_Change'] / df_final['Prev_Close']) * 100
            
            # Filter ke rentang tanggal asli yang dipilih user
            return df_final.loc[start_date:end_date]
        
        return None
    except Exception as e:
        st.sidebar.error(f"⚠️ Detail Error Saham: {str(e)}")
        return None

def crawl_and_analyze(keyword, start_date, end_date):
    try:
        googlenews = GoogleNews(lang='id', region='ID')
        # Filter tanggal menggunakan query agar lebih presisi di Google News
        search_query = f"{keyword} after:{start_date} before:{end_date}"
        googlenews.search(search_query)

        data_to_append = []
        for i in range(1, 6):
            googlenews.getpage(i)
            news = googlenews.results()
            if not news: continue
            df_temp = pd.DataFrame(news)
            data_to_append.append(df_temp)

        if not data_to_append: return None, None, None, None

        df = pd.concat(data_to_append, ignore_index=True).drop_duplicates(subset="title")
        df["title"] = df["title"].fillna("")
        df["desc"] = df["desc"].fillna("")
        
        df_texts = df.copy()
        df_texts["document"] = (df["title"] + " " + df["desc"]).apply(preprocess_text)
        long_string = " ".join(df_texts["document"].values)

        wordcloud = None
        most_common_words = None
        if long_string.strip():
            wordcloud = WordCloud(background_color="white", max_words=100).generate(long_string)
            most_common_words = Counter(long_string.split()).most_common(10)

        media_col = "media" if "media" in df.columns else "site" if "site" in df.columns else None
        top_media = df[media_col].value_counts().head(10) if media_col else pd.Series()

        return df_texts, wordcloud, most_common_words, top_media
    except Exception as e:
        st.sidebar.error(f"⚠️ Detail Error Berita: {str(e)}")
        return None, None, None, None

# --- UI STREAMLIT ---
st.set_page_config(page_title="Sentimen & Saham", layout="wide")

st.markdown("## 💹 Analisis Sentimen Berita & Pergerakan Saham")
st.write("Menganalisis korelasi antara pemberitaan media dan harga saham harian.")

# Sidebar Pengaturan
st.sidebar.header("⚙️ Konfigurasi")
ticker = st.sidebar.text_input("Simbol Saham (Yahoo Finance):", value="BBCA.JK")
keyword = st.sidebar.text_input("Kata Kunci Berita:", value="Bank Central Asia")

# Input Tanggal
c_start, c_end = st.sidebar.columns(2)
start_date = c_start.date_input("Mulai:", datetime.now() - timedelta(days=14))
end_date = c_end.date_input("Selesai:", datetime.now())

if st.sidebar.button("🚀 Jalankan Analisis"):
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    with st.spinner("Sedang menyinkronkan data..."):
        # Eksekusi Fungsi
        df_stock = get_stock_data(ticker, start_str, end_str)
        df_news, wc, common, media = crawl_and_analyze(keyword, start_str, end_str)

        tab1, tab2, tab3 = st.tabs(["📉 Pergerakan Saham", "📰 Daftar Berita", "🔠 Analisis Kata"])

        with tab1:
            if df_stock is not None and not df_stock.empty:
                st.subheader(f"Statistik Harga {ticker}")
                # Urutan kolom: Prev Close -> Close -> Change
                display_cols = ['Prev_Close', 'Close', 'Price_Change', 'Pct_Change (%)']
                st.dataframe(df_stock[display_cols].style.format("{:.2f}"), use_container_width=True)
                
                # Visualisasi harga
                st.line_chart(df_stock['Close'])
            else:
                st.error("Gagal mengambil data saham. Silakan cek 'Logs' di Streamlit Cloud atau coba mundurkan tanggal.")

        with tab2:
            if df_news is not None:
                st.subheader("Berita Terkait")
                st.dataframe(df_news[['date', 'title', 'media', 'link']], use_container_width=True, height=400)
                if not media.empty:
                    st.subheader("Media Paling Sering Meliput")
                    st.bar_chart(media)
            else:
                st.warning("Data berita tidak ditemukan untuk periode ini.")

        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                if wc:
                    st.subheader("Word Cloud")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
            with c2:
                if common:
                    st.subheader("Kata Kunci Dominan")
                    st.table(pd.DataFrame(common, columns=["Kata", "Frekuensi"]))
