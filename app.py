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

# --- KONFIGURASI ---
AV_API_KEY = "CYJG0OMG7PWSU1V9"

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
        # Menggunakan 'compact' karena limitasi akun gratis (100 data terakhir)
        data, meta_data = ts.get_daily(symbol=ticker, outputsize='compact')
        
        df = data.rename(columns={'4. close': 'Close'})
        df = df.sort_index()
        
        # Kalkulasi kolom tambahan untuk tabel
        df['Prev_Close'] = df['Close'].shift(1)
        df['Price_Change'] = df['Close'] - df['Prev_Close']
        df['Pct_Change (%)'] = (df['Price_Change'] / df['Prev_Close']) * 100
        
        # Filter berdasarkan rentang tanggal
        df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
        df_filtered = df.loc[(df.index >= start_date) & (df.index <= end_date)]
        
        return df_filtered
    except Exception as e:
        if "rate limit" in str(e).lower():
            st.sidebar.error("⚠️ Limit API tercapai. Tunggu 60 detik.")
        else:
            st.sidebar.error(f"⚠️ Error: {str(e)}")
        return None

def crawl_news(keyword, start_date, end_date):
    try:
        gn = GoogleNews(lang='id', region='ID')
        search_q = f"{keyword} after:{start_date} before:{end_date}"
        gn.search(search_q)
        
        res = []
        for i in range(1, 3):
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
st.markdown("## 💹 Dashboard Analisis Saham (Alpha Vantage API)")

# Sidebar
st.sidebar.header("⚙️ Pengaturan")
ticker_input = st.sidebar.text_input("Ticker Saham:", value="BBCA")
keyword = st.sidebar.text_input("Kata Kunci Berita:", value="Bank Central Asia")

c1, c2 = st.sidebar.columns(2)
start_d = c1.date_input("Mulai", datetime.now() - timedelta(days=20))
end_d = c2.date_input("Selesai", datetime.now())

if st.sidebar.button("🚀 Jalankan Analisis"):
    s_str, e_str = start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")
    
    with st.spinner("Sinkronisasi data..."):
        df_s = get_stock_data_av(ticker_input, s_str, e_str)
        df_n, wc, common = crawl_news(keyword, s_str, e_str)

        tab1, tab2 = st.tabs(["📉 Grafik & Harga", "📰 Berita & Kata"])

        with tab1:
            if df_s is not None and not df_s.empty:
                st.subheader(f"Statistik Harga {ticker_input}")
                # Tabel data lengkap
                st.dataframe(df_s[['Prev_Close', 'Close', 'Price_Change', 'Pct_Change (%)']].style.format("{:.2f}"), use_container_width=True)
                
                # Grafik HANYA harga Close
                st.subheader(f"Grafik Harga Penutupan (Close) {ticker_input}")
                st.line_chart(df_s['Close'])
            else:
                st.error("Data saham tidak ditemukan. Cek kembali ticker atau rentang tanggal.")

        with tab2:
            if df_n is not None:
                st.subheader("Berita Terkait")
                st.dataframe(df_n[['date', 'title', 'media', 'link']], use_container_width=True)
                if wc:
                    st.subheader("Analisis Kata")
                    fig, ax = plt.subplots()
                    ax.imshow(wc); ax.axis("off")
                    st.pyplot(fig)
            else:
                st.warning("Berita tidak ditemukan.")
