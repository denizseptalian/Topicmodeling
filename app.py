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
        # Ambil H-7 untuk mendapatkan Prev Close
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=7)
        
        # Download data dengan auto_adjust agar kolom lebih konsisten
        df_stock = yf.download(ticker, start=start_dt.strftime("%Y-%m-%d"), end=end_date, auto_adjust=True)
        
        if df_stock.empty:
            return None

        # --- FIX MULTIINDEX ---
        # yfinance terbaru sering kasih kolom (Close, Ticker). Kita ratakan jadi (Close) saja.
        if isinstance(df_stock.columns, pd.MultiIndex):
            df_stock.columns = df_stock.columns.get_level_values(0)

        df_final = df_stock.copy()
        
        # Pastikan kolom Close ada
        if 'Close' in df_final.columns:
            df_final['Prev_Close'] = df_final['Close'].shift(1)
            df_final['Price_Change'] = df_final['Close'] - df_final['Prev_Close']
            df_final['Pct_Change (%)'] = (df_final['Price_Change'] / df_final['Prev_Close']) * 100
            
            # Filter ke range asli
            return df_final.loc[start_date:end_date]
        return None
    except Exception as e:
        st.error(f"⚠️ Error yfinance: {str(e)}")
        return None

def crawl_and_analyze(keyword, start_date, end_date):
    try:
        googlenews = GoogleNews(lang='id', region='ID')
        # Gunakan filter after/before di query untuk akurasi rentang tanggal
        search_query = f"{keyword} after:{start_date} before:{end_date}"
        googlenews.search(search_query)

        data_to_append = []
        for i in range(1, 6): # Ambil 5 halaman awal
            googlenews.getpage(i)
            news = googlenews.results()
            if len(news) == 0: continue
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
        st.error(f"⚠️ Error Google News: {str(e)}")
        return None, None, None, None

# --- STREAMLIT UI ---
st.set_page_config(page_title="Analisis Sentimen & Saham", layout="wide")

st.markdown("### 📈 Analisis Sentimen Berita & Pergerakan Saham")

# Sidebar
st.sidebar.header("⚙️ Pengaturan")
ticker = st.sidebar.text_input("Simbol Saham (Yahoo Finance):", value="BBCA.JK")
keyword = st.sidebar.text_input("Kata Kunci Berita:", value="Bank Central Asia")

col_s, col_e = st.sidebar.columns(2)
start_date = col_s.date_input("Mulai:", datetime.now() - timedelta(days=7))
end_date = col_e.date_input("Selesai:", datetime.now())

if st.sidebar.button("Jalankan Analisis"):
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    with st.spinner("Mengolah data..."):
        # Ambil Data
        df_stock = get_stock_data(ticker, start_str, end_str)
        df_news, wc, common, media = crawl_and_analyze(keyword, start_str, end_str)

        tab1, tab2, tab3 = st.tabs(["📊 Pergerakan Saham", "📰 Berita", "🔍 Analisis Kata"])

        with tab1:
            if df_stock is not None and not df_stock.empty:
                st.subheader(f"Data Saham {ticker}")
                # Menampilkan kolom sesuai permintaan: Prev Close, Close, Price Change
                cols_to_show = ['Prev_Close', 'Close', 'Price_Change', 'Pct_Change (%)']
                st.dataframe(df_stock[cols_to_show].style.format("{:.2f}"))
                st.line_chart(df_stock['Close'])
            else:
                st.error("Gagal menarik data saham. Coba ganti ticker atau cek koneksi.")

        with tab2:
            if df_news is not None:
                st.subheader("Data Berita")
                st.dataframe(df_news[['date', 'title', 'media', 'link']], height=400)
                if not media.empty:
                    st.subheader("🏆 Media Teraktif")
                    st.bar_chart(media)
            else:
                st.warning("Berita tidak ditemukan.")

        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                if wc:
                    st.subheader("☁️ Word Cloud")
                    fig, ax = plt.subplots()
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
            with c2:
                if common:
                    st.subheader("📌 Kata Dominan")
                    st.table(pd.DataFrame(common, columns=["Kata", "Jumlah"]))
