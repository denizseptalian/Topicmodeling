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

logging.basicConfig(level=logging.INFO)

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
    
    # Preprocess the text data for LDA
    documents = df['title'].fillna('') + ' ' + df['desc'].fillna('')  # Combine title and description
    df_texts = pd.DataFrame(documents, columns=['document'])
    processed_docs = [doc.split() for doc in df_texts['document']]
    id2word = gensim.corpora.Dictionary(processed_docs)
    corpus = [id2word.doc2bow(doc) for doc in processed_docs]
    
    # Build LDA model
    num_topics = 3
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           random_state=0)
    
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

if keyword:
    try:
        # Perform crawling and analysis
        df_dominant_topic, lda_model, corpus, id2word, wordcloud = crawl_and_analyze(keyword)
        
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
        for idx, topic in enumerate(lda_model.print_topics()):
            st.write(f"Topic {idx + 1}")
            st.write(topic[1])
        
        # pyLDAvis visualization
        st.subheader("LDA Visualization")
        LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
        pyLDAvis.save_html(LDAvis_prepared, 'ldavis.html')
        with open('ldavis.html', 'r') as f:
            html_string = f.read()
        st.components.v1.html(html_string, width=1300, height=800)
        
    except Exception as e:
        logging.exception("An error occurred during processing.")
        st.error(f"An error occurred: {e}")
