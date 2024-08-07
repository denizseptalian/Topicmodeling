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

# Fungsi untuk prapemrosesan teks
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    text = stopword.remove(text)
    return text

# Fungsi untuk mengambil dan menganalisis data
def crawl_and_analyze(keyword):
    logging.info("Mulai pengambilan data dari Google News")
    start_time = time.time()
    
    googlenews = GoogleNews(lang='id', region='ID')
    googlenews.search(keyword)
    
    data_to_append = []
    for i in range(1, 2):  # Ambil hanya satu halaman untuk pengujian
        time.sleep(1)  # Kurangi waktu sleep
        googlenews.getpage(i)
        news = googlenews.results()[:5]  # Ambil hanya 5 dokumen per halaman
        if news:
            df_temp = pd.DataFrame(news)
            data_to_append.append(df_temp)
    
    if data_to_append:
        df = pd.concat(data_to_append, ignore_index=True)
    else:
        raise ValueError("Tidak ada data yang diambil dari Google News")
    
    end_time = time.time()
    logging.info(f"Pengambilan data selesai dalam {end_time - start_time:.2f} detik")

    if 'title' not in df.columns or 'desc' not in df.columns:
        raise KeyError("Kolom 'title' atau 'desc' yang diperlukan hilang dalam data yang diambil")

    documents = df['title'].fillna('') + ' ' + df['desc'].fillna('')
    df_texts = pd.DataFrame(documents, columns=['document'])
    df_texts['document'] = df_texts['document'].apply(preprocess_text)
    
    logging.info("Mulai prapemrosesan teks")
    start_time = time.time()
    
    processed_docs = [doc.split() for doc in df_texts['document']]
    id2word = gensim.corpora.Dictionary(processed_docs)
    corpus = [id2word.doc2bow(doc) for doc in processed_docs]
    
    end_time = time.time()
    logging.info(f"Prapemrosesan teks selesai dalam {end_time - start_time:.2f} detik")

    logging.info("Mulai pelatihan model LDA")
    start_time = time.time()
    
    num_topics = 2  # Gunakan topik yang lebih sedikit untuk pengujian
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           passes=1,  # Gunakan passes yang lebih sedikit untuk pengujian
                                           iterations=10,  # Gunakan iterasi yang lebih sedikit untuk pengujian
                                           workers=1,  # Gunakan hanya satu worker untuk menghindari overhead paralelisasi
                                           random_state=0)
    
    end_time = time.time()
    logging.info(f"Pelatihan model LDA selesai dalam {end_time - start_time:.2f} detik")

    df_dominant_topic = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=df_texts['document'].tolist(), original_df=df)
    
    long_string = ', '.join(df_texts['document'].values)
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue').generate(long_string)
    
    return df_dominant_topic, lda_model, corpus, id2word, wordcloud

# Fungsi untuk memformat topik per kalimat
def format_topics_sentences(ldamodel, corpus, texts, original_df):
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
    
    sent_topics_df = pd.concat([sent_topics_df, original_df[['media']].reset_index(drop=True)], axis=1)
    
    return sent_topics_df

# UI Streamlit
st.title("Pencarian Kata Kunci dan Analisis LDA")

keyword = st.text_input("Masukkan kata kunci untuk pencarian:")

if keyword:
    try:
        df_dominant_topic, lda_model, corpus, id2word, wordcloud = crawl_and_analyze(keyword)
        
        st.subheader("Dataframe Topik Dominan")
        st.dataframe(df_dominant_topic)
        
        st.subheader("Word Cloud")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        
        st.subheader("Topik LDA")
        word_counter = Counter()
        for idx, topic in enumerate(lda_model.print_topics()):
            st.write(f"Topik {idx + 1}")
            st.write(topic[1])
            
            words, probs = zip(*lda_model.show_topic(idx, topn=10))
            word_counter.update(words)
        
        common_words = word_counter.most_common(10)
        words, counts = zip(*common_words)
        plt.figure(figsize=(10, 5))
        plt.barh(words, counts)
        plt.xlabel("Jumlah")
        plt.title("10 Kata Teratas di Semua Topik")
        st.pyplot(plt)
        
        st.subheader("Visualisasi LDA")
        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
        pyLDAvis_html = pyLDAvis.prepared_data_to_html(vis)
        st.components.v1.html(pyLDAvis_html, height=1000, width=1250)
        
    except Exception as e:
        logging.exception("Terjadi kesalahan selama pemrosesan.")
        st.error(f"Terjadi kesalahan: {e}")
