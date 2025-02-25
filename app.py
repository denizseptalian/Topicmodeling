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
    # Remove irrelevant characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove stopwords
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    text = stopword.remove(text)
    return text

# Function to crawl and analyze data
def crawl_and_analyze(keyword):
    googlenews = GoogleNews(lang='id', region='ID')
    googlenews.search(keyword)
    
    # Collect data from multiple pages
    data_to_append = []
    for i in range(1, 11):
        googlenews.getpage(i)
        news = googlenews.results()
        df_temp = pd.DataFrame(news)
        data_to_append.append(df_temp)
    
    # Concatenate all the data into one DataFrame
    df = pd.concat(data_to_append, ignore_index=True)
    
    # Preprocess the text data
    documents = df['title'].fillna('') + ' ' + df['desc'].fillna('')  # Combine title and description
    df_texts = df.copy()
    df_texts['document'] = documents.apply(preprocess_text)
    
    # Generate word cloud
    long_string = ', '.join(df_texts['document'].values)
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue').generate(long_string)
    
    # Count most common words
    words = long_string.split()
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(10)
    
    # Analyze top 10 media sources
    top_media = df['media'].value_counts().head(10)
    
    return df_texts, wordcloud, most_common_words, top_media

# Streamlit UI Styling
st.set_page_config(page_title="Keyword Crawling Analysis", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    h1, h2, h3 {
        color: #2E3B55;
    }
    .stDataFrame {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 Keyword Crawling & Analysis")
st.write("Crawl berita dari Google News berdasarkan kata kunci dan analisis hasilnya dengan visualisasi yang menarik.")

# Input keyword
keyword = st.text_input("📝 Masukkan Kata Kunci untuk Crawling:", placeholder="Contoh: Teknologi, Politik, Ekonomi")

if keyword:
    try:
        # Perform crawling and analysis
        df_texts, wordcloud, most_common_words, top_media = crawl_and_analyze(keyword)
        
        # Display the dataframe
        st.subheader("📊 Data Berita yang Dikumpulkan")
        st.dataframe(df_texts.drop(columns=['date'], errors='ignore'), height=400)
        
        # Word cloud visualization
        st.subheader("☁️ Word Cloud")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
        # Display most common words
        st.subheader("📌 10 Kata Paling Sering Muncul")
        st.table(pd.DataFrame(most_common_words, columns=["Kata", "Jumlah"], index=range(1, 11)))
        
        # Display top 10 media sources
        st.subheader("🏆 10 Media Paling Aktif")
        st.table(pd.DataFrame(top_media, columns=["Media", "Jumlah Berita"]).reset_index(drop=True))
    
    except Exception as e:
        logging.exception("An error occurred during processing.")
        st.error(f"❌ Terjadi kesalahan: {e}")
