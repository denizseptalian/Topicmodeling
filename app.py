import pandas as pd
import matplotlib.pyplot as plt
from investiny import historical_data, search_assets
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
st.set_page_config(layout="wide", page_title="Analisis Saham Investiny")

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

def preprocess_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return stopword.remove(text)

# =========================
# STOCK DATA (INVESTINY VERSION)
# =========================
def get_stock_data_investing(symbol, start_date, end_date):
    try:
        # 1. Cari ID Asset di Investing.com
        results = search_assets(query=symbol)
        if not results:
            return None
        
        # Ambil ID pertama yang ditemukan (biasanya yang paling relevan)
        asset_id = results[0]["exchange_id"] 
        
        # 2. Tarik Data Historis
        # Format tanggal investiny: MM/DD/YYYY
        from_d = start_date.strftime('%m/%d/%Y')
        to_d = end_date.strftime('%m/%d/%Y')
        
        data = historical_data(investing_id=asset_id, from_date=from_d, to_date=to_d)
        
        if not data: return None
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Kalkulasi kolom tambahan
        df['Prev_Close'] = df['close'].shift(1)
        df['Price_Change'] = df['close'] - df['Prev_Close']
        df['Pct_Change (%)'] = (df['Price_Change'] / df['Prev_Close']) * 100
        
        # Rename agar konsisten dengan tampilan sebelumnya
        df = df.rename(columns={'close': 'Close'})
        
        return df.dropna(subset=['Close'])

    except Exception as e:
        st.error(f"Error Investiny: {e}")
        return None

# =========================
# NEWS DATA (RSS FEED)
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
st.title("💹 Dashboard Saham (Investiny Mode)")

st.sidebar.header("Konfigurasi")
# Investiny lebih suka nama simpel seperti "BBCA" daripada "BBCA.JK"
ticker_input = st.sidebar.text_input("Ticker (Contoh: BBCA, TLKM, ASII):", value="BBCA")
keyword_input = st.sidebar.text_input("Kata Kunci Berita:", value="Bank BCA")

col_s, col_e = st.sidebar.columns(2)
date_start = col_s.date_input("Mulai", datetime.now() - timedelta(days=20))
date_end = col_e.date_input("Selesai", datetime.now())

if st.sidebar.button("🚀 Jalankan Analisis"):
    with st.spinner("Mengambil data dari Investing.com..."):
        df_stock = get_stock_data_investing(ticker_input, date_start, date_end)
        df_news, wc, common, media = crawl_news_rss(keyword_input)

        t1, t2, t3 = st.tabs(["📉 Harga Saham", "📰 Daftar Berita", "📊 Analisis"])

        with t1:
            if df_stock is not None and not df_stock.empty:
                st.subheader(f"Data Harga {ticker_input} (Investing.com)")
                st.dataframe(df_stock[['Prev_Close', 'Close', 'Price_Change', 'Pct_Change (%)']].style.format("{:.2f}"), use_container_width=True)
                st.line_chart(df_stock["Close"])
            else:
                st.error("Data saham tidak ditemukan di Investiny. Coba masukkan nama perusahaan (misal: Bank Central Asia).")

        with t2:
            if df_news is not None:
                st.dataframe(df_news[["Tanggal", "Judul", "Sumber", "Link"]], use_container_width=True)

        with t3:
            if wc:
                fig, ax = plt.subplots()
                ax.imshow(wc); ax.axis("off")
                st.pyplot(fig)
                st.bar_chart(media)
