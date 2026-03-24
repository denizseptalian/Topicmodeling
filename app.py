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
st.set_page_config(layout="wide", page_title="Analisis Saham & Sentimen")

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

def preprocess_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return stopword.remove(text)

# =========================
# STOCK DATA (INVESTINY)
# =========================
def get_stock_data_investing(symbol, start_date, end_date):
    try:
        # Cari ID aset (investing_id) berdasarkan ticker
        search_results = search_assets(query=symbol)
        if not search_results:
            return None
        
        # Ambil ID dari hasil pertama (pastikan exchange-nya relevan)
        asset_id = search_results[0]["exchange_id"]
        
        # Format tanggal untuk investiny (MM/DD/YYYY)
        from_date = start_date.strftime('%m/%d/%Y')
        to_date = end_date.strftime('%m/%d/%Y')
        
        data = historical_data(investing_id=asset_id, from_date=from_date, to_date=to_date)
        
        if not data: return None
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Perhitungan teknis sederhana
        df['Prev_Close'] = df['close'].shift(1)
        df['Price_Change'] = df['close'] - df['Prev_Close']
        df['Pct_Change (%)'] = (df['Price_Change'] / df['Prev_Close']) * 100
        
        df = df.rename(columns={'close': 'Close'})
        return df.dropna(subset=['Close'])

    except Exception as e:
        st.sidebar.error(f"Error Saham: {e}")
        return None

# =========================
# NEWS DATA (RSS)
# =========================
def crawl_news_rss(keyword):
    try:
        url = f"https://news.google.com/rss/search?q={keyword.replace(' ', '%20')}&hl=id&gl=ID&ceid=ID:id"
        feed = feedparser.parse(url)
        
        if not feed.entries: return None, None, None, None

        data = []
        for e in feed.entries:
            data.append({
                "Tanggal": e.get("published", ""),
                "Judul": e.title,
                "Sumber": e.get("source", {}).get("title", ""),
                "Link": e.link
            })
        
        df = pd.DataFrame(data)
        df["doc"] = df["Judul"].apply(preprocess_text)
        all_text = " ".join(df["doc"])
        
        wc = WordCloud(background_color="white", width=800, height=400).generate(all_text) if all_text.strip() else None
        common = Counter(all_text.split()).most_common(10)
        media = df["Sumber"].value_counts().head(10)
        
        return df, wc, common, media
    except:
        return None, None, None, None

# =========================
# USER INTERFACE
# =========================
st.title("💹 Dashboard Analisis Saham & Sentimen")
st.caption("Data: Investing.com (via Investiny) | Berita: Google News RSS")

st.sidebar.header("Konfigurasi")
ticker_input = st.sidebar.text_input("Ticker (Contoh: BBCA, TLKM):", value="BBCA")
keyword_input = st.sidebar.text_input("Kata Kunci Berita:", value="Bank BCA")

c1, c2 = st.sidebar.columns(2)
start_d = c1.date_input("Mulai", datetime.now() - timedelta(days=20))
end_d = c2.date_input("Selesai", datetime.now())

if st.sidebar.button("🚀 Jalankan Analisis"):
    with st.spinner("Sedang memproses data..."):
        df_s = get_stock_data_investing(ticker_input, start_d, end_d)
        df_n, wc, common, media = crawl_news_rss(keyword_input)

        tab1, tab2, tab3 = st.tabs(["📉 Pergerakan Saham", "📰 Daftar Berita", "📊 Analisis Kata"])

        with tab1:
            if df_s is not None and not df_s.empty:
                st.subheader(f"Data Historis {ticker_input}")
                st.dataframe(df_s[['Prev_Close', 'Close', 'Price_Change', 'Pct_Change (%)']].style.format("{:.2f}"), use_container_width=True)
                st.line_chart(df_s["Close"])
            else:
                st.error("Data saham tidak ditemukan. Coba gunakan nama perusahaan jika ticker gagal.")

        with tab2:
            if df_n is not None:
                st.success(f"Ditemukan {len(df_n)} berita.")
                st.dataframe(df_n[["Tanggal", "Judul", "Sumber", "Link"]], use_container_width=True)
            else:
                st.warning("Berita tidak ditemukan.")

        with tab3:
            if wc:
                st.subheader("WordCloud Berita")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
                
                col_left, col_right = st.columns(2)
                with col_left:
                    st.subheader("Kata Terbanyak")
                    st.table(pd.DataFrame(common, columns=["Kata", "Jumlah"]))
                with col_right:
                    st.subheader("Top Media")
                    st.bar_chart(media)
