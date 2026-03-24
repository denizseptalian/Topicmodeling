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
from bs4 import BeautifulSoup

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
# STOCK DATA (SCRAPER BEAUTIFULSOUP)
# =========================
def get_stock_data_scraping(ticker, start_date, end_date):
    try:
        if not ticker.endswith(".JK"): ticker = f"{ticker}.JK"
        
        # URL Halaman Riwayat Harga Yahoo Finance
        url = f"https://finance.yahoo.com/quote/{ticker}/history"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            st.error(f"Server Yahoo menolak akses (Code: {response.status_code}). Coba lagi nanti.")
            return None
            
        # Parsing HTML tabel harga
        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.find('table', {'data-test': 'historical-prices'})
        
        if not table:
            # Fallback jika struktur HTML berubah sedikit
            table = soup.find('table', {'class': 'W(100%) M(0)'})

        df = pd.read_html(str(table))[0]
        
        # Bersihkan data (hapus baris dividen atau data kosong)
        df = df[df['Open'].str.contains("Dividend|Stock Split") == False]
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Pastikan kolom angka benar-benar numerik
        cols = ['Open', 'High', 'Low', 'Close*', 'Adj Close**', 'Volume']
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.sort_values('Date').set_index('Date')
        
        # Kalkulasi kolom tambahan
        df = df.rename(columns={'Close*': 'Close'})
        df['Prev_Close'] = df['Close'].shift(1)
        df['Price_Change'] = df['Close'] - df['Prev_Close']
        df['Pct_Change (%)'] = (df['Price_Change'] / df['Prev_Close']) * 100
        
        # Filter range tanggal
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        return df.loc[mask]

    except Exception as e:
        st.sidebar.error(f"Gagal memproses tabel: {e}")
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
        df["doc_clean"] = df["Judul"].apply(preprocess_text)
        txt = " ".join(df["doc_clean"])
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
st.info("Metode: BeautifulSoup Scraper (Bypass 401) | Berita: RSS")

st.sidebar.header("Filter Analisis")
ticker_in = st.sidebar.text_input("Ticker (Contoh: BBCA, ASII):", value="BBCA")
keyword_in = st.sidebar.text_input("Keyword Berita:", value="Bank BCA")

c1, c2 = st.sidebar.columns(2)
start_d = c1.date_input("Mulai", datetime.now() - timedelta(days=20))
end_d = c2.date_input("Selesai", datetime.now())

if st.sidebar.button("🚀 Ambil Data"):
    with st.spinner("Membaca tabel harga dari website..."):
        df_s = get_stock_data_scraping(ticker_in, start_d, end_d)
        df_n, wc, common, media = crawl_news_rss(keyword_in)

        t1, t2, t3 = st.tabs(["📉 Harga Saham", "📰 Berita", "📊 Analisis"])

        with t1:
            if df_s is not None and not df_s.empty:
                st.subheader(f"Data Harga {ticker_in} (Rp)")
                st.dataframe(df_s[['Prev_Close', 'Close', 'Price_Change', 'Pct_Change (%)']].style.format("{:.2f}"), use_container_width=True)
                st.line_chart(df_s["Close"])
            else:
                st.warning("Tabel harga tidak ditemukan atau kosong. Coba muat ulang halaman.")

        with t2:
            if df_n is not None:
                st.dataframe(df_n[["Tanggal", "Judul", "Sumber", "Link"]], use_container_width=True)

        with t3:
            if wc:
                fig, ax = plt.subplots(figsize=(10,5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
                st.bar_chart(media)
