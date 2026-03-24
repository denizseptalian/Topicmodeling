import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from wordcloud import WordCloud
from collections import Counter
import re
import feedparser
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# OPTIONAL ALPHA
try:
    from alpha_vantage.timeseries import TimeSeries
    ALPHA_OK = True
except:
    ALPHA_OK = False

# ================= CONFIG =================
AV_API_KEY = "YQNUKAH419JA2RYV"
HEADERS = {"User-Agent": "Mozilla/5.0"}

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

st.set_page_config(layout="wide")

# ================= KURS =================
@st.cache_data(ttl=3600)
def get_kurs():
    try:
        r = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=IDR").json()
        return r['rates']['IDR']
    except:
        return 15500

# ================= PREPROCESS =================
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return stopword.remove(text)

# ================= YAHOO SAFE =================
def get_yahoo_safe(symbol, start, end):

    def to_unix(d):
        return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.JK"

    try:
        r = requests.get(url, headers=HEADERS, params={
            "interval": "1d",
            "period1": to_unix(start),
            "period2": to_unix(end + timedelta(days=1))
        }, timeout=10)

        data = r.json()
        result = data.get("chart", {}).get("result")

        if not result:
            return None

        ts = result[0]["timestamp"]
        close = result[0]["indicators"]["quote"][0]["close"]

        rows = []
        for t, c in zip(ts, close):
            if c is not None:
                rows.append({
                    "Date": datetime.fromtimestamp(t).date(),
                    "Close": c
                })

        df = pd.DataFrame(rows)

        kurs = get_kurs()

        df['Prev_Close'] = df['Close'].shift(1)
        df['Price_Change'] = df['Close'] - df['Prev_Close']
        df['Pct_Change (%)'] = (df['Price_Change'] / df['Prev_Close']) * 100
        df['Close_IDR'] = df['Close'] * kurs

        return df

    except:
        return None

# ================= ALPHA =================
def get_alpha(symbol, start, end):

    if not ALPHA_OK:
        return None

    try:
        ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=symbol, outputsize='compact')

        df = data.rename(columns={'4. close': 'Close'})
        df.index = pd.to_datetime(df.index)

        df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]

        df = df.reset_index().rename(columns={"index":"Date"})
        df['Date'] = pd.to_datetime(df['Date']).dt.date

        kurs = get_kurs()

        df['Prev_Close'] = df['Close'].shift(1)
        df['Price_Change'] = df['Close'] - df['Prev_Close']
        df['Pct_Change (%)'] = (df['Price_Change'] / df['Prev_Close']) * 100
        df['Close_IDR'] = df['Close'] * kurs

        return df

    except:
        return None

# ================= NEWS FULL ANALYSIS =================
@st.cache_data(ttl=300)
def get_news_full(keyword, start, end):

    all_data = []

    # ambil banyak halaman via RSS (loop hari)
    date_range = pd.date_range(start, end)

    for d in date_range:
        url = f"https://news.google.com/rss/search?q={keyword}+after:{d.date()}+before:{(d+timedelta(days=1)).date()}&hl=id&gl=ID&ceid=ID:id"
        feed = feedparser.parse(url)

        for e in feed.entries:
            all_data.append({
                "date": pd.to_datetime(e.published).date(),
                "title": e.title,
                "desc": getattr(e, "summary", ""),
                "source": e.source.title if hasattr(e, "source") else ""
            })

    if not all_data:
        return None, None, None, None

    df = pd.DataFrame(all_data).drop_duplicates(subset="title")

    # ================= TEXT PROCESS =================
    df["doc"] = (df["title"] + " " + df["desc"]).apply(preprocess_text)

    text = " ".join(df["doc"])

    wc, common = None, None

    if text.strip():
        wc = WordCloud(background_color="white", max_words=5000).generate(text)
        common = Counter(text.split()).most_common(10)

    # ================= TOP MEDIA =================
    top_media = df["source"].value_counts().head(10)

    # ================= GROUP PER HARI =================
    df_daily = df.groupby("date")["title"].apply(lambda x: " | ".join(x)).reset_index()
    df_daily.rename(columns={"date":"Date"}, inplace=True)

    return df, df_daily, wc, common, top_media

# ================= STYLE =================
def color(val):
    if pd.isna(val):
        return ""
    return "color: green" if val > 0 else "color: red"

# ================= UI =================
st.title("💹 Analisis Sentimen Berita Ekonomi pada Google News dan Pengaruhnya terhadap Volatilitas serta Pergerakan Intraday Harga Saham")

source = st.selectbox("Sumber Saham", ["Yahoo (IDX)", "Alpha (Global)"])
ticker = st.text_input("Ticker", "BBCA")
keyword = st.text_input("Keyword Berita", "Bank BCA")

c1, c2 = st.columns(2)
start = c1.date_input("Start", datetime.now() - timedelta(days=30))
end = c2.date_input("End", datetime.now())

if st.button("🚀 Jalankan Analisis"):

    # SAHAM
    if source == "Yahoo (IDX)":
        df_s = get_yahoo_safe(ticker, start, end)
    else:
        df_s = get_alpha(ticker, start, end)

    # BERITA FULL
    df_news, df_daily, wc, common, media = get_news_full(keyword, start, end)

    tab1, tab2 = st.tabs(["📉 Saham + Merge", "📰 Analisis Berita"])

    # ================= TAB 1 =================
    with tab1:
        if df_s is not None and df_daily is not None:

            df_merge = pd.merge(df_s, df_daily, on="Date", how="left")

            cols = [
                'Date','Prev_Close','Close','Close_IDR',
                'Price_Change','Pct_Change (%)','title'
            ]

            styled = df_merge[cols].style.format({
                "Prev_Close": "{:.2f}",
                "Close": "{:.2f}",
                "Close_IDR": "Rp {:,.0f}",
                "Price_Change": "{:.2f}",
                "Pct_Change (%)": "{:.2f}"
            }).applymap(color, subset=['Price_Change','Pct_Change (%)'])

            st.dataframe(styled, use_container_width=True)

            st.subheader("📈 Grafik Harga (Rupiah)")
            st.line_chart(df_merge.set_index("Date")["Close_IDR"])

        else:
            st.error("❌ Data saham / berita gagal")

    # ================= TAB 2 =================
    with tab2:
        if df_news is not None:

            st.subheader("📊 Data Berita Lengkap")
            st.dataframe(df_news, use_container_width=True)

            if wc:
                st.subheader("☁️ WordCloud")
                fig, ax = plt.subplots()
                ax.imshow(wc)
                ax.axis("off")
                st.pyplot(fig)

            if common:
                st.subheader("📌 Top 10 Kata")
                st.table(pd.DataFrame(common, columns=["Kata","Jumlah"]))

            if not media.empty:
                st.subheader("🏆 Top Media")
                st.table(
                    media.reset_index()
                    .rename(columns={"index":"Media","source":"Jumlah"})
                )

        else:
            st.warning("⚠️ Tidak ada berita ditemukan")
