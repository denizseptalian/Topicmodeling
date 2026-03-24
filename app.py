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

logging.basicConfig(level=logging.INFO)

# Initialize stopword remover once
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# --- Fungsi Preprocessing ---
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = stopword.remove(text)
    return text

# --- Fungsi Crawl Berita dengan Rentang Tanggal ---
def crawl_and_analyze(keyword, start_date, end_date):
    # Format tanggal untuk Google News: MM/DD/YYYY atau gunakan after/before di query
    googlenews = GoogleNews(lang='id', region='ID')
    
    # Menambahkan filter tanggal ke query
    search_query = f"{keyword} after:{start_date} before:{end_date}"
    googlenews.search(search_query)

    data_to_append = []
    # Mengambil 5 halaman pertama (bisa ditambah ke 10 jika perlu)
    for i in range(1, 6):
        try:
            googlenews.getpage(i)
            news = googlenews.results()
            if len(news) == 0: continue
            df_temp = pd.DataFrame(news)
            df_temp = df_temp.drop_duplicates(subset="title")
            data_to_append.append(df_temp)
        except Exception as e:
            logging.warning(f"Gagal mengambil page {i}: {e}")

    if len(data_to_append) == 0:
        return None, None, None, None

    df = pd.concat(data_to_append, ignore_index=True)
    df["title"] = df["title"].fillna("")
    df["desc"] = df["desc"].fillna("")
    
    # Gabungkan teks untuk analisis WordCloud
    df_texts = df.copy()
    df_texts["document"] = (df["title"] + " " + df["desc"]).apply(preprocess_text)
    long_string = " ".join(df_texts["document"].values)

    wordcloud = None
    most_common_words = None
    if long_string.strip() != "":
        wordcloud = WordCloud(background_color="white", max_words=100).generate(long_string)
        words = long_string.split()
        most_common_words = Counter(words).most_common(10)

    media_col = "media" if "media" in df.columns else "site" if "site" in df.columns else None
    top_media = df[media_col].value_counts().head(10) if media_col else pd.Series()

    return df_texts, wordcloud, most_common_words, top_media

# --- Fungsi Ambil Data Saham ---
def get_stock_data(ticker, start_date, end_date):
    # Ambil H-5 untuk mendapatkan Prev Close
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=7)
    df_stock = yf.download(ticker, start=start_dt.strftime("%Y-%m-%d"), end=end_date)
    
    if df_stock.empty:
        return None

    # Handle MultiIndex yfinance
    if isinstance(df_stock.columns, pd.MultiIndex):
        df_stock.columns = df_stock.columns.get_level_values(0)

    df_stock['Prev_Close'] = df_stock['Close'].shift(1)
    df_stock['Price_Change'] = df_stock['Close'] - df_stock['Prev_Close']
    df_stock['Pct_Change (%)'] = (df_stock['Price_Change'] / df_stock['Prev_Close']) * 100
    
    # Filter kembali ke range yang dipilih
    df_stock = df_stock.loc[start_date:end_date]
    return df_stock

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Sentimen Berita & Saham", layout="wide")

st.title("📈 Analisis Sentimen Berita & Pergerakan Saham")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Pengaturan Analisis")
ticker = st.sidebar.text_input("Simbol Saham (Yahoo Finance):", value="BBCA.JK")
keyword = st.sidebar.text_input("Kata Kunci Berita:", value="Bank Central Asia")

# Date Input
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Mulai:", datetime.now() - timedelta(days=7))
end_date = col2.date_input("Selesai:", datetime.now())

if st.sidebar.button("Jalankan Analisis"):
    if keyword and ticker:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        with st.spinner("Sedang memproses data berita dan saham..."):
            try:
                # 1. Analisis Berita
                df_texts, wordcloud, most_common_words, top_media = crawl_and_analyze(keyword, start_str, end_str)
                
                # 2. Analisis Saham
                df_stock = get_stock_data(ticker, start_str, end_str)

                # --- TAB TAMPILAN ---
                tab1, tab2, tab3 = st.tabs(["📊 Pergerakan Saham", "📰 Berita & Sentimen", "🔍 Analisis Kata"])

                with tab1:
                    if df_stock is not None:
                        st.subheader(f"Data Pergerakan Saham {ticker}")
                        # Menampilkan kolom sesuai urutan permintaan Anda
                        st.dataframe(df_stock[['Prev_Close', 'Close', 'Price_Change', 'Pct_Change (%)']].style.format("{:.2f}"))
                        
                        # Plot Harga
                        st.line_chart(df_stock['Close'])
                    else:
                        st.error("Gagal mengambil data saham. Pastikan ticker benar.")

                with tab2:
                    if df_texts is not None:
                        st.subheader("Kumpulan Berita Google News")
                        # Sederhanakan dataframe berita
                        st.dataframe(df_texts[['date', 'title', 'media', 'link']], height=400)
                        
                        if not top_media.empty:
                            st.subheader("🏆 Media Paling Aktif")
                            st.bar_chart(top_media)
                    else:
                        st.warning("Tidak ada berita ditemukan.")

                with tab3:
                    c1, c2 = st.columns(2)
                    with c1:
                        if wordcloud:
                            st.subheader("☁️ Word Cloud")
                            fig, ax = plt.subplots()
                            ax.imshow(wordcloud, interpolation="bilinear")
                            ax.axis("off")
                            st.pyplot(fig)
                    with c2:
                        if most_common_words:
                            st.subheader("📌 Kata Kunci Dominan")
                            st.table(pd.DataFrame(most_common_words, columns=["Kata", "Jumlah"]))

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
                logging.exception(e)
    else:
        st.warning("Mohon isi kata kunci dan simbol saham.")
