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
import time

logging.basicConfig(level=logging.INFO)

# Function to preprocess the text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    text = stopword.remove(text)
    return text

# Function to fetch news data from multiple pages
@st.cache(show_spinner=False)
def fetch_news_data(keyword, num_pages):
    googlenews = GoogleNews(lang='id', region='ID')
    all_news = []

    for i in range(1, num_pages + 1):
        googlenews.search(keyword)
        googlenews.getpage(i)
        all_news.extend(googlenews.results())
    
    return pd.DataFrame(all_news)

# Function to crawl and analyze data
@st.cache(show_spinner=False)
def crawl_and_analyze(keyword, num_pages=5):
    df = fetch_news_data(keyword, num_pages)

    # Preprocess the text data for LDA
    documents = df['title'].fillna('') + ' ' + df['desc'].fillna('')
    df_texts = pd.DataFrame(documents, columns=['document'])
    df_texts['document'] = df_texts['document'].apply(preprocess_text)

    processed_docs = [doc.split() for doc in df_texts['document']]
    id2word = gensim.corpora.Dictionary(processed_docs)
    corpus = [id2word.doc2bow(doc) for doc in processed_docs]

    # Build LDA model
    num_topics = 10
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=0, passes=2, workers=2)

    # Create a dataframe for dominant topic
    df_dominant_topic = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=df_texts['document'].tolist())

    # Generate word cloud
    long_string = ', '.join(df_texts['document'].values)
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue').generate(long_string)

    return df_dominant_topic, lda_model, corpus, id2word, wordcloud

# Function to format topics per sentence
def format_topics_sentences(ldamodel, corpus, texts):
    sent_topics_df = pd.DataFrame()

    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = pd.concat([sent_topics_df, pd.DataFrame([[int(topic_num), round(prop_topic, 4), topic_keywords]])], ignore_index=True)
            else:
                break

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents.reset_index(drop=True)], axis=1)
    return sent_topics_df

# Streamlit UI
st.title("Keyword Crawling and LDA Analysis")

# Input keyword
keyword = st.text_input("Enter a keyword for crawling:")
num_pages = st.slider("Number of pages to crawl:", min_value=1, max_value=10, value=5)

if keyword:
    try:
        start_time = time.time()
        
        with st.spinner('Crawling and analyzing data...'):
            # Perform crawling and analysis
            df_dominant_topic, lda_model, corpus, id2word, wordcloud = crawl_and_analyze(keyword, num_pages)
        
        # Display the dataframe
        st.subheader("Dominant Topic DataFrame")
        st.dataframe(df_dominant_topic)
        
        # Word cloud visualization
        st.subheader("Word Cloud")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        
        # LDA topics visualization
        st.subheader("LDA Topics")
        word_counter = Counter()
        for idx, topic in enumerate(lda_model.print_topics()):
            st.write(f"Topic {idx + 1}")
            st.write(topic[1])
            
            # Update word counter with the words from each topic
            words, probs = zip(*lda_model.show_topic(idx, topn=10))
            word_counter.update(words)
        
        # Plot the top 10 words across all topics
        common_words = word_counter.most_common(10)
        words, counts = zip(*common_words)
        plt.figure(figsize=(10, 5))
        plt.barh(words, counts)
        plt.xlabel("Counts")
        plt.title("Top 10 words across all topics")
        st.pyplot(plt)
        
        # pyLDAvis visualization
        st.subheader("LDA Visualization")
        LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
        pyLDAvis.save_html(LDAvis_prepared, 'ldavis.html')
        with open('ldavis.html', 'r') as f:
            html_string = f.read()
        st.components.v1.html(html_string, width=1300, height=800)
        
        end_time = time.time()
        st.success(f"Processing completed in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        logging.exception("An error occurred during processing.")
        st.error(f"An error occurred: {e}")
