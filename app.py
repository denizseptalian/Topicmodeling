import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GoogleNews import GoogleNews
import streamlit as st
import logging
import time

logging.basicConfig(level=logging.INFO)

# Function to fetch news data from multiple pages
@st.cache_data(show_spinner=False)
def fetch_news_data(keyword, num_pages):
    googlenews = GoogleNews(lang='id', region='ID')
    all_news = []

    for i in range(1, num_pages + 1):
        googlenews.clear()
        googlenews.search(keyword)
        googlenews.getpage(i)
        all_news.extend(googlenews.results())
    
    return pd.DataFrame(all_news)

# Function to process and analyze news data
@st.cache_data(show_spinner=False)
def analyze_news(keyword, num_pages=5):
    df = fetch_news_data(keyword, num_pages)
    
    if df.empty:
        return None, None, df
    
    # Convert date column to datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])  # Remove rows with invalid dates
    df['date'] = df['date'].dt.date  # Keep only date, remove time info
    
    # Count articles per day
    trend_data = df.groupby('date').size().reset_index(name='count')
    
    # Count articles per publisher
    publisher_data = df['media'].value_counts().reset_index()
    publisher_data.columns = ['Publisher', 'Count']
    
    return trend_data, publisher_data, df

# Streamlit UI
st.title("📊 Analisis Tren Waktu dan Klasifikasi Penerbit Berita Google News")

# Input keyword
keyword = st.text_input("Masukkan kata kunci untuk crawling:")
num_pages = st.slider("Jumlah halaman yang akan dicrawl:", min_value=1, max_value=10, value=5)

if keyword:
    try:
        start_time = time.time()
        
        with st.spinner('Crawling dan menganalisis data...'):
            trend_data, publisher_data, df = analyze_news(keyword, num_pages)
        
        if trend_data is None:
            st.error("Tidak ada berita ditemukan untuk kata kunci ini.")
        else:
            # Display full dataset
            st.subheader("📰 Data Berita yang Dicrawling")
            st.dataframe(df)
            
            # Display trend data
            st.subheader("📈 Tren Waktu Berita")
            plt.figure(figsize=(10, 5))
            plt.plot(trend_data['date'], trend_data['count'], marker='o', linestyle='-')
            plt.xlabel("Tanggal")
            plt.ylabel("Jumlah Berita")
            plt.title("Tren Waktu Berita")
            plt.xticks(rotation=45)
            st.pyplot(plt)
            
            # Display publisher data
            st.subheader("🏢 Kategorisasi Penerbit Berita")
            st.dataframe(publisher_data)
            
            # Visualize top publishers
            st.subheader("🔝 Penerbit Berita Teratas")
            plt.figure(figsize=(10, 5))
            plt.barh(publisher_data['Publisher'][:10], publisher_data['Count'][:10], color='steelblue')
            plt.xlabel("Jumlah Berita")
            plt.ylabel("Penerbit")
            plt.title("Top 10 Penerbit Berita")
            st.pyplot(plt)
        
        end_time = time.time()
        st.success(f"✅ Proses selesai dalam {end_time - start_time:.2f} detik")
        
    except Exception as e:
        logging.exception("Terjadi kesalahan saat memproses data.")
        st.error(f"⚠ Terjadi kesalahan: {e}")

