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
    
    # Convert date column to datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
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
    
    # Analyze trending news over time
    trend_data = df['date'].dt.date.value_counts().sort_index()
    
    # Analyze top 10 media sources
    top_media = df['media'].value_counts().head(10)
    
    return df_texts, wordcloud, most_common_words, trend_data, top_media

# Streamlit UI
st.title("Keyword Crawling Analysis")

# Input keyword
keyword = st.text_input("Enter a keyword for crawling:")

if keyword:
    try:
        # Perform crawling and analysis
        df_texts, wordcloud, most_common_words, trend_data, top_media = crawl_and_analyze(keyword)
        
        # Display the dataframe
        st.subheader("Crawled Data")
        st.dataframe(df_texts)
        
        # Word cloud visualization
        st.subheader("Word Cloud")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        
        # Display most common words
        st.subheader("Most Common Words")
        st.write(pd.DataFrame(most_common_words, columns=["Word", "Count"]))
        
        # Display news trend over time
        st.subheader("Trending News Over Time")
        plt.figure(figsize=(10, 5))
        trend_data.plot(kind='bar')
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.title("News Trend Over Time")
        st.pyplot(plt)
        
        # Display top 10 media sources
        st.subheader("Top 10 Media Sources")
        st.write(pd.DataFrame(top_media, columns=["Media", "Article Count"]))
    
    except Exception as e:
        logging.exception("An error occurred during processing.")
        st.error(f"An error occurred: {e}")
