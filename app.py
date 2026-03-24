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

# --- KONFIGURASI API KEY (TETAP SAMA) ---
AV_API_KEY = "CYJG0OMG7PWSU1V9" # API Key Anda sudah terpasang

# Inisialisasi Sastrawi
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

def preprocess_text(text):
    if pd.isna(text) or text == "": return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return stopword.remove(text)

def get_stock_data_av(ticker, start_date, end_date):
    try:
        ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
        
        # PERBAIKAN: Gunakan 'compact' (untuk akun gratis Alpha Vantage, mengambil 100 data terakhir)
        data, meta_data = ts.get_daily(symbol=ticker, outputsize='compact')
        
        # Mapping kolom Alpha Vantage: '4. close' -> 'Close'
        df = data.rename(columns={'4. close': 'Close'})
        
        # Pastikan data terurut dari tanggal lama ke baru (PENTING untuk Prev Close)
        df = df.sort_index()
        
        # Perhitungan data saham (Prev Close, Change)
        df['Prev_Close'] = df['Close'].shift(1)
        df['Price_Change'] = df['Close'] - df['Prev_Close']
        df['Pct_Change (%)'] = (df['Price_Change'] / df['Prev_Close']) * 100
        
        # Konversi index ke string untuk filtering
        df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
        
        # Filter berdasarkan rentang tanggal user (Start Date s/d End Date)
        # Ingat: 'compact' hanya mengambil 100 hari terakhir, pastikan rentang tanggal masuk.
        df_filtered = df.loc[(df.index >= start_date) & (df.index <= end_date)]
        
        if df_filtered.empty:
            st.sidebar.warning("Data kosong. Pastikan rentang tanggal masuk dalam 100 hari terakhir (compact mode).")
            return None
            
        return df_filtered
    except Exception as e:
        # Jika error karena limit API (5x per menit)
        if "rate limit" in str(e).lower():
            st.sidebar.error("⚠️ Limit API tercapai. Tunggu 1 menit lalu klik tombol lagi.")
        else:
            st.sidebar.error(f"⚠️ Alpha Vantage Error: {str(e)}")
        return None

def crawl_news(keyword, start_date, end_date):
    try:
        gn = GoogleNews(lang='id', region='ID')
        # Gunakan query agar Google News memfilter tanggal
        search_q = f"{keyword} after:{start_date} before:{end_date}"
        gn.search(search_q)
        
        res = []
        for i in range(1, 4):
            gn.getpage(i)
            if gn.results(): res.append(pd.DataFrame(gn.results()))
        
        if not res: return None, None, None
        
        df = pd.concat(res).drop_duplicates(subset="title")
        df["doc"] = (df["title"].fillna("") + " " + df["desc"].fillna("")).apply(preprocess_text)
        
        txt = " ".join(df["doc"].values)
        wc = WordCloud(background_color="white").generate(txt) if txt.strip() else None
        common = Counter(txt.split()).most_common(10)
        
        return df, wc, common
    except:
        return None, None, None

# --- UI STREAMLIT ---
st.set_page_config(page_title="Analisis Sentimen & Saham", layout="wide")
st.markdown("## 💹 Dashboard Analisis Saham (Alpha Vantage Official API)")
st.info("Aplikasi ini menghubungkan sentimen berita harian dengan fluktuasi harga saham di bursa.")

# Sidebar Pengaturan
st.sidebar.header("⚙️ Konfigurasi")
ticker_input = st.sidebar.text_input("Ticker Saham (Contoh: BBCA.JK atau IDX:BBCA):", value="BBCA")
keyword = st.sidebar.text_input("Kata Kunci Berita:", value="Bank Central Asia")

col1, col2 = st.sidebar.columns(2)
start_d = col1.date_input("Mulai", datetime.now() - timedelta(days=30))
end_d = col2.date_input("Selesai", datetime.now())

if st.sidebar.button("🚀 Jalankan Analisis"):
    s_str, e_str = start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")
    
    with st.spinner("Menarik data dari API Resmi..."):
        df_s = get_stock_data_av(ticker_input, s_str, e_str)
        df_n, wc, common = crawl_news(keyword, s_str, e_str)

        tab1, tab2 = st.tabs(["📉 Grafik & Data Saham", "📰 Berita & WordCloud"])

        # --- TAB 1: GRAFIK & DATA SAHAM ---
        with tab1:
            if df_s is not None and not df_s.empty:
                st.subheader(f"Statistik Harga Saham {ticker_input}")
                
                # Menampilkan tabel harga (Urutan: Prev Close, Close, Change)
                show_cols = ['Prev_Close', 'Close', 'Price_Change', 'Pct_Change (%)']
                st.dataframe(df_s[show_cols].style.format("{:.2f}"), use_container_width=True)
                
                # Menampilkan Grafik Pergerakan Harga Close
                st.subheader(f"Grafik Pergerakan Harga {ticker_input} (Close)")
                # 'Close' harus dalam format index yang valid untuk line_chart (pandas dataframe)
                st.line_chart(df_s['Close'])
                
            else:
                st.error("Gagal menampilkan data saham. Pastikan rentang tanggal masuk dalam 100 hari perdagangan terakhir (Mode Compact).")

        # --- TAB 2: BERITA & WORDCLOUD ---
        with tab2:
            if df_n is not None:
                st.subheader("Kumpulan Berita Google News")
                # Tampilkan kolom penting berita
                st.dataframe(df_n[['date', 'title', 'media', 'link']], use_container_width=True)
                
                if wc:
                    st.subheader("Visualisasi Kata Terbanyak (WordCloud)")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ Tidak cukup teks untuk membuat WordCloud.")
            else:
                st.warning("⚠️ Berita tidak ditemukan untuk rentang tanggal tersebut.")
