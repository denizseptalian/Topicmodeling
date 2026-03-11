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

# Initialize stopword remover once
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# Function to preprocess the text
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = stopword.remove(text)
    return text


# Function to crawl and analyze data
def crawl_and_analyze(keyword):

    googlenews = GoogleNews(lang='id', region='ID')
    googlenews.search(keyword)

    data_to_append = []

    for i in range(1, 11):

        try:
            googlenews.getpage(i)
            news = googlenews.results()

            if len(news) == 0:
                continue

            df_temp = pd.DataFrame(news)

            # Avoid duplicates
            df_temp = df_temp.drop_duplicates(subset="title")

            data_to_append.append(df_temp)

        except Exception as e:
            logging.warning(f"Gagal mengambil page {i}: {e}")

    if len(data_to_append) == 0:
        return None, None, None, None

    df = pd.concat(data_to_append, ignore_index=True)

    # Handle missing columns safely
    df["title"] = df["title"] if "title" in df.columns else ""
    df["desc"] = df["desc"] if "desc" in df.columns else ""

    df["title"] = df["title"].fillna("")
    df["desc"] = df["desc"].fillna("")

    documents = df["title"] + " " + df["desc"]

    df_texts = df.copy()
    df_texts["document"] = documents.apply(preprocess_text)

    long_string = " ".join(df_texts["document"].values)

    # Handle empty text for wordcloud
    wordcloud = None
    most_common_words = None

    if long_string.strip() != "":

        wordcloud = WordCloud(
            background_color="white",
            max_words=5000,
            contour_width=3,
            contour_color="steelblue"
        ).generate(long_string)

        words = long_string.split()
        word_counts = Counter(words)
        most_common_words = word_counts.most_common(10)

    # Detect media column
    media_col = None

    if "media" in df.columns:
        media_col = "media"
    elif "site" in df.columns:
        media_col = "site"

    if media_col:
        top_media = df[media_col].value_counts().head(10)
    else:
        top_media = pd.Series()

    return df_texts, wordcloud, most_common_words, top_media


# ===============================
# STREAMLIT UI
# ===============================

st.set_page_config(
    page_title="Keyword Crawling Analysis",
    layout="wide"
)

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

keyword = st.text_input(
    "📝 Masukkan Kata Kunci untuk Crawling:",
    placeholder="Contoh: Teknologi, Politik, Ekonomi"
)

if keyword:

    with st.spinner("Mengambil data berita..."):

        try:

            df_texts, wordcloud, most_common_words, top_media = crawl_and_analyze(keyword)

            if df_texts is None:
                st.error("❌ Tidak ada berita yang ditemukan untuk kata kunci ini.")

            else:

                st.subheader("📊 Data Berita yang Dikumpulkan")

                st.dataframe(
                    df_texts.drop(columns=["date"], errors="ignore"),
                    height=400
                )

                if wordcloud:

                    st.subheader("☁️ Word Cloud")

                    fig, ax = plt.subplots(figsize=(10,5))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")

                    st.pyplot(fig)

                else:
                    st.warning("⚠️ Tidak cukup teks untuk membuat WordCloud.")

                if most_common_words:

                    st.subheader("📌 10 Kata Paling Sering Muncul")

                    st.table(
                        pd.DataFrame(
                            most_common_words,
                            columns=["Kata", "Jumlah"],
                            index=range(1, len(most_common_words)+1)
                        )
                    )

                if not top_media.empty:

                    st.subheader("🏆 10 Media Paling Aktif")

                    st.table(
                        pd.DataFrame(top_media)
                        .reset_index()
                        .rename(columns={"index":"Media", top_media.name:"Jumlah Berita"})
                    )

                else:
                    st.warning("⚠️ Tidak ada data media yang ditemukan.")

        except Exception as e:

            logging.exception("Error during processing")

            st.error(f"❌ Terjadi kesalahan: {e}")
