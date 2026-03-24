import pandas as pd
import matplotlib.pyplot as plt
from stocker import predict # Kita gunakan fungsi predict untuk ambil data historis
import streamlit as st
from wordcloud import WordCloud
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
from datetime import datetime, timedelta
import feedparser

# =========================
# CONFIG & INITIALIZATION
# =========================
st.set_page_config(layout="wide", page_title="Dashboard Analisis Tesis")

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

def preprocess_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return stopword.remove(text)

# =========================
# STOCK DATA (STOCKER VERSION)
# =========================
def get_stock_data_stocker(ticker):
    try:
        # Stocker otomatis mencari data dari berbagai sumber
        # Jika input BBCA, kita coba arahkan ke BBCA.JK
        if not ticker.endswith(".JK"):
            symbol = f"{ticker}.JK"
        else:
            symbol = ticker
            
        from stocker.get_data import GetData
        df = GetData(symbol).get_data()
        
        if df is None or df.empty:
            return None
            
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Kalkulasi kolom tambahan
        df['Prev_Close'] = df['Close'].shift(1)
        df['Price_Change'] = df['Close'] - df['Prev_Close']
        df['Pct_Change (%)'] = (df['Price_Change'] / df['Prev_Close']) * 100
        
        return df
    except Exception as e:
        st.sidebar.error(f"Error Stocker: {e}")
        return None

# =========================
# NEWS DATA (RSS)
# =========================
def crawl_news_rss(keyword):
    try:
        url = f"https://news.google.com/rss/search?q={keyword.replace(' ', '%20')}&hl=id&gl=ID&ceid=ID:id"
        feed = feedparser.parse(url)
        if not feed.entries: return None, None, None, None

        data = [{"Tanggal": e.get("published", ""), "Judul": e.title, "Sumber": e.get("source", {}).get("title", ""), "Link": e.link} for e in feed.entries]
        df = pd.DataFrame(data)
        df["doc"] = df["Judul"].apply(preprocess_text)
        txt = " ".join(df["doc"])
        wc = WordCloud(background_color="white", width=800, height=400).generate(txt) if txt.strip() else None
        common = Counter(txt.split()).most_common(10)
        media = df["Sumber"].value_counts().head(10)
        return df, wc, common, media
    except:
        return None, None, None, None

# =========================
# UI
# =========================
st.title("💹 Dashboard Analisis Saham & Sentimen")
st.info("Menggunakan Stocker Engine untuk menghindari blokir server.")

st.sidebar.header("Parameter")
ticker_in = st.sidebar.text_input("Ticker (Contoh: BBCA, ASII):", value="BBCA")
keyword_in = st.sidebar.text_input("Kata Kunci Berita:", value="Bank BCA")

if st.sidebar.button("🚀 Ambil Data"):
    with st.spinner("Menghubungkan ke engine data..."):
        df_s = get_stock_data_stocker(ticker_in)
        df_n, wc, common, media = crawl_news_rss(keyword_in)

        t1, t2, t3 = st.tabs(["📉 Harga Saham", "📰 Berita", "📊 Analisis"])

        with t1:
            if df_s is not None:
                st.subheader(f"Data Harga {ticker_in} (Terbaru)")
                # Tampilkan 20 data terakhir agar tidak terlalu panjang
                st.dataframe(df_s[['Prev_Close', 'Close', 'Price_Change', 'Pct_Change (%)']].tail(20).style.format("{:.2f}"), use_container_width=True)
                st.line_chart(df_s["Close"])
            else:
                st.error("Engine Stocker gagal mendapatkan data. Server sedang sangat sibuk.")

        with t2:
            if df_n is not None:
                st.dataframe(df_n[["Tanggal", "Judul", "Sumber", "Link"]], use_container_width=True)

        with t3:
            if wc:
                fig, ax = plt.subplots(figsize=(10,5))
                ax.imshow(wc); ax.axis("off")
                st.pyplot(fig)
                st.bar_chart(media)
