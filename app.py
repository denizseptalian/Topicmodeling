import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GoogleNews import GoogleNews
import streamlit as st
from wordcloud import WordCloud
import logging
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re

logging.basicConfig(level=logging.INFO)

# Function to preprocess the text
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    return stopword.remove(text)

# Function to crawl and analyze data
def crawl_and_analyze(keyword):
    googlenews = GoogleNews(lang='id', region='ID')
    googlenews.search(keyword)
    
    # Collect data from multiple pages
    data_to_append = []
    for i in range(1, 6):  # Ambil lebih sedikit halaman untuk efisiensi
        googlenews.getpage(i)
        news = googlenews.results()
        df_temp = pd.DataFrame(news)
        data_to_append.append(df_temp)
    
    # Gabungkan semua hasil scraping
    df = pd.concat(data_to_append, ignore_index=True)

    # **Cek apakah hasil scraping kosong**
    if df.empty:
        return None, None, None, None

    # **Gunakan .get() untuk menghindari error jika kolom tidak ditemukan**
    df['title'] = df.get('title', '').fillna('')
    df['desc'] = df.get('desc', '').fillna('')
    df['media'] = df.get('media', '').fillna('Tidak Diketahui')

    # **Gabungkan title dan desc untuk preprocessing**
    df['document'] = (df['title'] + ' ' + df['desc']).apply(preprocess_text)

    # **Buat word cloud**
    long_string = ', '.join(df['document'].values)
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue').generate(long_string)

    # **Hitung kata paling sering muncul**
    words = long_string.split()
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(10)

    # **Hitung 10 Media Paling Aktif**
    top_media = df['media'].value_counts().head(10)

    return df, wordcloud, most_common_words, top_media

# Streamlit UI
st.set_page_config(page_title="Keyword Crawling Analysis", layout="wide")
st.title("🔍 Keyword Crawling & Analysis")
st.write("Crawl berita dari Google News berdasarkan kata kunci dan analisis hasilnya.")

# Input keyword
keyword = st.text_input("📝 Masukkan Kata Kunci untuk Crawling:", placeholder="Contoh: Teknologi, Politik, Ekonomi")

if keyword:
    try:
        df, wordcloud, most_common_words, top_media = crawl_and_analyze(keyword)

        if df is None:
            st.error("❌ Tidak ada berita yang ditemukan untuk kata kunci ini.")
        else:
            # **Tampilkan data berita**
            st.subheader("📊 Data Berita yang Dikumpulkan")
            st.dataframe(df.drop(columns=['date'], errors='ignore'), height=400)

            # **Word cloud**
            st.subheader("☁️ Word Cloud")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

            # **10 Kata Paling Sering Muncul**
            st.subheader("📌 10 Kata Paling Sering Muncul")
            st.table(pd.DataFrame(most_common_words, columns=["Kata", "Jumlah"], index=range(1, 11)))

            # **10 Media Paling Aktif**
            st.subheader("🏆 10 Media Paling Aktif")
            if not top_media.empty:
                st.table(pd.DataFrame(top_media).reset_index().rename(columns={"index": "Media", "media": "Jumlah Berita"}))
            else:
                st.write("Tidak ada data media yang tersedia.")

    except Exception as e:
        logging.exception("An error occurred during processing.")
        st.error(f"❌ Terjadi kesalahan: {e}")
