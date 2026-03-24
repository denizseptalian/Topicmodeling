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

# Inisialisasi Logging
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
        # Ambil rentang lebih luas untuk mendapatkan 'Previous Close'
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=10)
        start_str = start_dt.strftime("%Y-%m-%d")
        
        # Penyamaran User-Agent agar tidak diblokir Streamlit Cloud
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Download data dengan auto_adjust=True agar kolom konsisten
        df_stock = yf.download(
            ticker, 
            start=start_str, 
            end=end_date, 
            progress=False,
            auto_adjust=True,
            headers=headers
        )
        
        if df_stock is None or df_stock.empty or len(df_stock) < 2:
            return None

        # --- FIX MULTIINDEX ---
        # Menghapus level ticker pada kolom (misal: ('Close', 'BBCA.JK') -> 'Close')
        if isinstance(df_stock.columns, pd.MultiIndex):
            df_stock.columns = df_stock.columns.get_level_values(0)

        df_final = df_stock.copy()
        
        # Cari kolom harga penutupan (Close)
        target_col = 'Close' if 'Close' in df_final.columns else df_final.columns[0]

        # Logika Perhitungan sesuai permintaan User
        df_final['Prev_Close'] = df_final[target_col].shift(1)
        df_final['Price_Change'] = df_final[target_col] - df_final['Prev_Close']
        df_final['Pct_Change (%)'] = (df_final['Price_Change'] / df_final['Prev_Close']) * 100
        
        # Filter ke rentang tanggal yang diinginkan (Start Date s/d End Date)
        # Pastikan format index adalah string untuk filter yang akurat
        df_final.index = df_final.index.strftime('%Y-%m-%d')
        df_result = df_final.loc[df_final.index >= start_date]
        
        return df_result

    except Exception as e:
        st.sidebar.error(f"⚠️ Detail Error Saham: {str(e)}")
        return None

def crawl_and_analyze(keyword, start_date, end_date):
    try:
        googlenews = GoogleNews(lang='id', region='ID')
        search_query = f"{keyword} after:{start_date} before:{end_date}"
        googlenews.search(search_query)

        data_to_append = []
        for i in range(1, 6):
            googlenews.getpage(i)
            news = googlenews.results()
            if not news: continue
            data_to_append.append(pd.DataFrame(news))

        if not data_to_append: return None, None, None, None

        df = pd.concat(data_to_append, ignore_index=True).drop_duplicates(subset="title")
        df_texts = df.copy()
        df_texts["document"] = (df["title"].fillna("") + " " + df["desc"].fillna("")).apply(preprocess_text)
        
        long_string = " ".join(df_texts["document"].values)
        wordcloud = WordCloud(background_color="white", max_words=100).generate(long_string) if long_string.strip() else None
        common = Counter(long_string.split()).most_common(10) if long_string.strip() else None
        
        media_col = "media" if "media" in df.columns else "site" if "site" in df.columns else None
        top_media = df[media_col].value_counts().head(10) if media_col else pd.Series()

        return df_texts, wordcloud, common, top_media
    except Exception as e:
        st.sidebar.error(f"⚠️ Detail Error Berita: {str(e)}")
        return None, None, None, None

# --- Tampilan Utama Streamlit ---
st.set_page_config(page_title="Sentimen & Saham", layout="wide")

st.markdown("## 💹 Analisis Sentimen Berita & Pergerakan Saham")
st.info("Aplikasi ini menghubungkan sentimen berita harian dengan fluktuasi harga saham di bursa.")

# Sidebar
st.sidebar.header("⚙️ Konfigurasi")
ticker = st.sidebar.text_input("Simbol Saham (Yahoo Finance):", value="BBCA.JK")
keyword = st.sidebar.text_input("Kata Kunci Berita:", value="Bank Central Asia")

c1, c2 = st.sidebar.columns(2)
start_date = c1.date_input("Mulai:", datetime.now() - timedelta(days=14))
end_date = c2.date_input("Selesai:", datetime.now())

if st.sidebar.button("🚀 Jalankan Analisis"):
    s_str = start_date.strftime("%Y-%m-%d")
    e_str = end_date.strftime("%Y-%m-%d")

    with st.spinner("Sinkronisasi data sedang berlangsung..."):
        df_stock = get_stock_data(ticker, s_str, e_str)
        df_news, wc, common, media = crawl_and_analyze(keyword, s_str, e_str)

        t1, t2, t3 = st.tabs(["📉 Pergerakan Saham", "📰 Daftar Berita", "🔠 Analisis Kata"])

        with t1:
            if df_stock is not None and not df_stock.empty:
                st.subheader(f"Statistik Perubahan Harga {ticker}")
                # Kolom: Prev_Close, Close, Price_Change
                cols = ['Prev_Close', 'Close', 'Price_Change', 'Pct_Change (%)']
                st.dataframe(df_stock[cols].style.format("{:.2f}"), use_container_width=True)
                st.line_chart(df_stock['Close'])
            else:
                st.error("Gagal menarik data saham. Coba mundurkan tanggal atau cek Logs.")

        with t2:
            if df_news is not None:
                st.subheader("Berita Terkini")
                st.dataframe(df_news[['date', 'title', 'media', 'link']], use_container_width=True)
                if not media.empty:
                    st.subheader("Media Paling Aktif")
                    st.bar_chart(media)
            else:
                st.warning("Berita tidak ditemukan.")

        with t3:
            ca, cb = st.columns(2)
            if wc:
                with ca:
                    st.subheader("Word Cloud")
                    fig, ax = plt.subplots()
                    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                    st.pyplot(fig)
            if common:
                with cb:
                    st.subheader("Kata Kunci Dominan")
                    st.table(pd.DataFrame(common, columns=["Kata", "Jumlah"]))
