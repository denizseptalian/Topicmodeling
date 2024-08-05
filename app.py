import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GoogleNews import GoogleNews
import streamlit as st
from wordcloud import WordCloud
import gensim
import pyLDAvis.gensim_models
import pyLDAvis
import logging
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO)

# Function to preprocess the text
def preprocess_text(texts):
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    
    def preprocess_batch(text_batch):
        return [stopword.remove(re.sub(r'[^a-zA-Z0-9\s]', '', text)) for text in text_batch]
    
    # Use multiprocessing for faster preprocessing
    with Pool() as pool:
        texts = pool.map(preprocess_batch, np.array_split(texts, pool._processes))
    return [item for sublist in texts for item in sublist]

# Function to crawl and analyze data
def crawl_and_analyze(keyword):
    googlenews = GoogleNews(lang='id', region='ID')
    googlenews.search(keyword)
    
    # Collect data from multiple pages
    data_to_append = []
    for i in range(1, 6):  # Limit to 5 pages for testing
        googlenews.getpage(i)
        news = googlenews.results()
        df_temp = pd.DataFrame(news)
        data_to_append.append(df_temp)
    
    # Concatenate all the data into one DataFrame
    df = pd.concat(data_to_append, ignore_index=True)
    
    # Preprocess the text data for LDA
    documents = df['title'].fillna('') + ' ' + df['desc'].fillna('')  # Combine title and description
    df_texts = pd.DataFrame(documents, columns=['document'])
    df_texts['document'] = preprocess_text(df_texts['document'])
    
    # Create and fit the LDA model using scikit-learn
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df_texts['document'])
    
    num_topics = 10
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(X)
    
    # Create a dataframe for dominant topic
    def format_topics_sentences(lda_model, X, vectorizer):
        topic_keywords = []
        for idx, topic in enumerate(lda_model.components_):
            words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
            topic_keywords.append(f"Topic {idx}: " + " ".join(words))
        return topic_keywords

    df_dominant_topic = format_topics_sentences(lda, X, vectorizer)
    
    # Generate word cloud
    long_string = ', '.join(df_texts['document'].values)
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue').generate(long_string)
    
    return df_dominant_topic, lda, vectorizer, wordcloud

# Streamlit UI
st.title("Keyword Crawling and LDA Analysis")

# Input keyword
keyword = st.text_input("Enter a keyword for crawling:")

if keyword:
    try:
        # Perform crawling and analysis
        df_dominant_topic, lda_model, vectorizer, wordcloud = crawl_and_analyze(keyword)
        
        # Display the dataframe
        st.subheader("Dominant Topic DataFrame")
        st.write(df_dominant_topic)
        
        # Word cloud visualization
        st.subheader("Word Cloud")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        
        # LDA topics visualization
        st.subheader("LDA Topics")
        word_counter = Counter()
        for idx, topic in enumerate(lda_model.components_):
            st.write(f"Topic {idx + 1}")
            st.write(" ".join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]))
            
            # Update word counter with the words from each topic
            words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()]
            word_counter.update(words)
        
        # Plot the top 10 words across all topics
        common_words = word_counter.most_common(10)
        words, counts = zip(*common_words)
        plt.figure(figsize=(10, 5))
        plt.barh(words, counts)
        plt.xlabel("Counts")
        plt.title("Top 10 words across all topics")
        st.pyplot(plt)
        
        # pyLDAvis visualization (if using gensim LDA model)
        st.subheader("LDA Visualization")
        if isinstance(lda_model, gensim.models.LdaMulticore):
            LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
            pyLDAvis.save_html(LDAvis_prepared, 'ldavis.html')
            with open('ldavis.html', 'r') as f:
                html_string = f.read()
            st.components.v1.html(html_string, width=1300, height=800)
        
    except Exception as e:
        logging.exception("An error occurred during processing.")
        st.error(f"An error occurred: {e}")
