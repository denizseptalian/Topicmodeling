import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
from datetime import datetime, timedelta
import feedparser
import requests

# =========================
# CONFIG & INITIALIZATION
# =========================
st.set_page_config(layout="wide", page_title="Dashboard Saham & Sentimen")

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

def preprocess_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return stopword.remove(text)

# =========================
# STOCK DATA (SCRAPER VERSION)
# =========================
def get_stock_data_final(ticker, start_date, end_date):
    try:
        # Kita gunakan Yahoo Finance via URL langsung (CSV Export)
        # Ini lebih sulit diblokir daripada API library
        end_ts = int(datetime.combine(end_date, datetime.min.time()).timestamp())
        start_ts = int(datetime.combine(start_date - timedelta(days=10), datetime.min.time()).timestamp())
        
        # Jika user input BBCA, kita tambahkan .JK otomatis
        if not ticker.endswith(".JK"):
            ticker = f"{ticker}.JK"
            
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_ts}&period2={end_ts}&interval=1d&events=history&includeAdjustedClose=true"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            st.sidebar.error(f"Gagal akses server (Code: {response.status_code})")
            return None
            
        import io
        df = pd.read_csv(io.StringIO(response.text))
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        
        # Kalkulasi
        df['Prev_Close'] = df['Close'].shift(1)
        df['Price_Change'] = df['Close'] - df['Prev_Close']
        df['Pct_Change (%)'] = (df['Price_Change'] / df['Prev_Close']) * 100
        
        return df.loc[df.index >= pd.to_datetime(start_date)]

    except Exception as e:
        st.sidebar.error(f"Error Scraper: {e}")
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
st.caption("Mode: Direct Access (Anti-403) | Berita: Google RSS")

st.sidebar.header("Konfigurasi")
ticker_input = st.sidebar.text_input("Ticker (Contoh: BBCA, ASII, TLKM):", value="BBCA")
keyword_input = st.sidebar.text_input("Kata Kunci Berita:", value="Bank BCA")

c1, c2 = st.sidebar.columns(2)
start_d = c1.date_input("Mulai", datetime.now() - timedelta(days=20))
end_d = c2.date_input("Selesai", datetime.now())

if st.sidebar.button("🚀 Jalankan Analisis"):
    with st.spinner("Menembus pertahanan server..."):
        df_s = get_stock_data_final(ticker_input, start_d, end_d)
        df_n, wc, common, media = crawl_news_rss(keyword_input)

        tab1, tab2, tab3 = st.tabs(["📉 Grafik Saham", "📰 Daftar Berita", "📊 Visualisasi"])

        with tab1:
            if df_s is not None and not df_s.empty:
                st.subheader(f"Data Harga {ticker_input} (Rp)")
                st.dataframe(df_s[['Prev_Close', 'Close', 'Price_Change', 'Pct_Change (%)']].style.format("{:.2f}"), use_container_width=True)
                st.line_chart(df_s["Close"])
            else:
                st.error("Gagal menarik data saham. Server sedang memproteksi diri. Coba lagi dalam 1 menit.")

        with tab2:
            if df_n is not None:
                st.dataframe(df_news_display := df_n[["Tanggal", "Judul", "Sumber", "Link"]], use_container_width=True)

        with tab3:
            if wc:
                fig, ax = plt.subplots(figsize=(10,5))
                ax.imshow(wc); ax.axis("off")
                st.pyplot(fig)
                st.bar_chart(media)
