import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from GoogleNews import GoogleNews
import streamlit as st
from wordcloud import WordCloud
import logging
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
import requests
from datetime import datetime, timedelta

# --- SETTING ANTARMUKA ---
st.set_page_config(page_title="Sentimen & Saham", layout="wide")

# Inisialisasi Sastrawi
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

def preprocess_text(text):
    if pd.isna(text): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return stopword.remove(text)

def get_stock_data(ticker, start_date, end_date):
    try:
        # Mundurkan tanggal untuk Prev Close
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=12)
        
        # LOGIKA ANTI-BLOCK: Menggunakan Session dan User-Agent Browser Asli
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://finance.yahoo.com',
            'Referer': 'https://finance.yahoo.com'
        })

        # Download data
        df_stock = yf.download(
            ticker, 
            start=start_dt.strftime("%Y-%m-%d"), 
            end=end_date, 
            progress=False,
            auto_adjust=True,
            session=session # Gunakan session yang sudah diberi header browser
        )
        
        if df_stock is None or df_stock.empty:
            return None

        # Fix MultiIndex (yfinance v0.2+)
        if isinstance(df_stock.columns, pd.MultiIndex):
            df_stock.columns = df_stock.columns.get_level_values(0)

        df_final = df_stock.copy()
        target_col = 'Close' if 'Close' in df_final.columns else df_final.columns[0]

        # Kalkulasi
        df_final['Prev_Close'] = df_final[target_col].shift(1)
        df_final['Price_Change'] = df_final[target_col] - df_final['Prev_Close']
        df_final['Pct_Change (%)'] = (df_final['Price_Change'] / df_final['Prev_Close']) * 100
        
        # Filter ke range user
        df_final.index = pd.to_datetime(df_final.index).strftime('%Y-%m-%d')
        return df_final.loc[df_final.index >= start_date]

    except Exception as e:
        st.sidebar.error(f"⚠️ Detail Error: {str(e)}")
        return None

def crawl_and_analyze(keyword, start_date, end_date):
    try:
        googlenews = GoogleNews(lang='id', region='ID')
        search_query = f"{keyword} after:{start_date} before:{end_date}"
        googlenews.search(search_query)
        
        data_to_append = []
        for i in range(1, 4):
            googlenews.getpage(i)
            news = googlenews.results()
            if news: data_to_append.append(pd.DataFrame(news))
        
        if not data_to_append: return None, None, None, None
        
        df = pd.concat(data_to_append, ignore_index=True).drop_duplicates(subset="title")
        df["document"] = (df["title"].fillna("") + " " + df["desc"].fillna("")).apply(preprocess_text)
        
        long_string = " ".join(df["document"].values)
        wc = WordCloud(background_color="white").generate(long_string) if long_string.strip() else None
        common = Counter(long_string.split()).most_common(10)
        top_media = df["media"].value_counts().head(10) if "media" in df.columns else pd.Series()
        
        return df, wc, common, top_media
    except:
        return None, None, None, None

# --- UI STREAMLIT ---
st.markdown("## 💹 Analisis Sentimen & Pergerakan Saham")

st.sidebar.header("⚙️ Konfigurasi")
ticker = st.sidebar.text_input("Simbol Saham:", value="BBCA.JK")
keyword = st.sidebar.text_input("Kata Kunci:", value="Bank Central Asia")

c1, c2 = st.sidebar.columns(2)
start_date = c1.date_input("Mulai:", datetime.now() - timedelta(days=14))
end_date = c2.date_input("Selesai:", datetime.now())

if st.sidebar.button("🚀 Jalankan Analisis"):
    s_str, e_str = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    
    with st.spinner("Mengambil data..."):
        df_stock = get_stock_data(ticker, s_str, e_str)
        df_news, wc, common, media = crawl_and_analyze(keyword, s_str, e_str)

        t1, t2, t3 = st.tabs(["📉 Saham", "📰 Berita", "🔠 Kata"])

        with t1:
            if df_stock is not None and not df_stock.empty:
                st.subheader(f"Data Harga {ticker}")
                st.dataframe(df_stock[['Prev_Close', 'Close', 'Price_Change', 'Pct_Change (%)']].style.format("{:.2f}"))
                st.line_chart(df_stock['Close'])
            else:
                st.error("Server Yahoo memblokir permintaan. Coba mundurkan tanggal atau tunggu beberapa menit.")

        with t2:
            if df_news is not None:
                st.dataframe(df_news[['date', 'title', 'media', 'link']], use_container_width=True)
            else:
                st.warning("Berita tidak ditemukan.")

        with t3:
            if common:
                st.table(pd.DataFrame(common, columns=["Kata", "Jumlah"]))
