# ================= IMPORT =================
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from wordcloud import WordCloud
from collections import Counter
import re, feedparser, urllib.parse

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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

# ================= SMART KEYWORD =================
def smart_keyword(keyword, ticker):
    if not keyword or keyword.strip() == "":
        keyword = ticker

    keyword = str(keyword).replace("\n"," ").replace("\r"," ")
    keyword = " ".join(keyword.split())

    suggestions = [
        keyword,
        f"saham {keyword}",
        f"{keyword} Indonesia",
        f"{keyword} berita",
        f"{keyword} stock"
    ]

    return keyword, suggestions, urllib.parse.quote(keyword)

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
    if pd.isna(text): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return stopword.remove(text)

# ================= SENTIMENT =================
positive_words = {"naik","untung","laba","positif","menguat","tumbuh","baik"}
negative_words = {"turun","rugi","negatif","melemah","anjlok","buruk"}

def sentiment_score(text):
    score = 0
    for w in text.split():
        if w in positive_words: score += 1
        if w in negative_words: score -= 1
    return score

def sentiment_label(score):
    if score > 0: return "Positif"
    if score < 0: return "Negatif"
    return "Netral"

# ================= YAHOO SAFE =================
def get_yahoo(symbol, start, end, is_indo=True):

    def to_unix(d):
        return int(datetime(d.year,d.month,d.day,tzinfo=timezone.utc).timestamp())

    sym = f"{symbol}.JK" if is_indo else symbol
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"

    try:
        r = requests.get(url, headers=HEADERS, params={
            "interval":"1d",
            "period1":to_unix(start),
            "period2":to_unix(end+timedelta(days=1))
        })

        data = r.json()
        result = data.get("chart",{}).get("result")

        if not result: return None

        ts = result[0]["timestamp"]
        close = result[0]["indicators"]["quote"][0]["close"]

        df = pd.DataFrame({
            "Date":[datetime.fromtimestamp(t).date() for t in ts],
            "Close":close
        }).dropna()

        df['Prev_Close'] = df['Close'].shift(1)
        df['Price_Change'] = df['Close'] - df['Prev_Close']
        df['Pct_Change (%)'] = (df['Price_Change']/df['Prev_Close'])*100

        if not is_indo:
            kurs = get_kurs()
            df['Close_IDR'] = df['Close'] * kurs

        return df

    except:
        return None

# ================= ALPHA =================
def get_alpha(symbol, start, end):

    if not ALPHA_OK: return None

    try:
        ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
        data,_ = ts.get_daily(symbol=symbol)

        df = data.rename(columns={'4. close':'Close'})
        df.index = pd.to_datetime(df.index)

        df = df.loc[(df.index>=pd.to_datetime(start)) & (df.index<=pd.to_datetime(end))]
        df = df.reset_index().rename(columns={"index":"Date"})
        df['Date'] = pd.to_datetime(df['Date']).dt.date

        kurs = get_kurs()
        df['Prev_Close'] = df['Close'].shift(1)
        df['Price_Change'] = df['Close'] - df['Prev_Close']
        df['Pct_Change (%)'] = (df['Price_Change']/df['Prev_Close'])*100
        df['Close_IDR'] = df['Close'] * kurs

        return df

    except:
        return None

# ================= NEWS =================
@st.cache_data(ttl=300)
def get_news(keyword_encoded, start, end):

    data = []

    for d in pd.date_range(start, end):
        url = f"https://news.google.com/rss/search?q={keyword_encoded}+after:{d.date()}+before:{(d+timedelta(days=1)).date()}&hl=id&gl=ID&ceid=ID:id"
        feed = feedparser.parse(url)

        for e in feed.entries:
            data.append({
                "date": pd.to_datetime(e.published).date(),
                "title": e.title,
                "desc": getattr(e,"summary",""),
                "source": e.source.title if hasattr(e,"source") else ""
            })

    if not data: return None,None,None,None,None

    df = pd.DataFrame(data).drop_duplicates(subset="title")

    df["doc"] = (df["title"]+" "+df["desc"]).apply(preprocess_text)

    df["sentiment_score"] = df["doc"].apply(sentiment_score)
    df["sentiment_label"] = df["sentiment_score"].apply(sentiment_label)

    text = " ".join(df["doc"])

    wc, common = None, None
    if text.strip():
        wc = WordCloud(background_color="white").generate(text)
        common = Counter(text.split()).most_common(10)

    media = df["source"].value_counts().head(10)

    df_daily = df.groupby("date")["title"].apply(lambda x:" | ".join(x)).reset_index()
    df_daily.rename(columns={"date":"Date"}, inplace=True)

    sent_daily = df.groupby("date")["sentiment_score"].mean().reset_index()
    sent_daily.rename(columns={"date":"Date","sentiment_score":"Sentiment"}, inplace=True)

    df_daily = pd.merge(df_daily, sent_daily, on="Date", how="left")

    return df, df_daily, wc, common, media

# ================= LSTM =================
def lstm_predict(df):

    data = df[['Close']].dropna().values
    if len(data) < 15: return None

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X,y=[],[]
    for i in range(10,len(data_scaled)):
        X.append(data_scaled[i-10:i])
        y.append(data_scaled[i])

    X,y=np.array(X),np.array(y)

    model = Sequential([
        LSTM(50,input_shape=(X.shape[1],1)),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X,y,epochs=5,verbose=0)

    pred = model.predict(X)
    pred = scaler.inverse_transform(pred)

    df_pred = df.iloc[10:].copy()
    df_pred['Prediksi'] = pred

    return df_pred

# ================= STYLE =================
def color(val):
    if pd.isna(val): return ""
    return "color: green" if val>0 else "color: red"

# ================= UI =================
st.title("💹 Analisis Sentimen Berita Ekonomi pada Google News dan Pengaruhnya terhadap Volatilitas serta Pergerakan Intraday Harga Saham")

market = st.selectbox("Jenis Saham", ["Indonesia (IDX)", "Global"])
source = st.selectbox("Sumber Data", ["Yahoo", "Alpha"])

ticker = st.text_input("Ticker", "BBCA")
keyword_input = st.text_input("Keyword Berita")

kw, suggestions, kw_encoded = smart_keyword(keyword_input, ticker)

st.caption("💡 Rekomendasi keyword:")
st.write(suggestions)

start = st.date_input("Start", datetime.now()-timedelta(days=30))
end = st.date_input("End", datetime.now())

if st.button("RUN"):

    is_indo = "Indonesia" in market

    # saham
    df_s = get_yahoo(ticker,start,end,is_indo) if source=="Yahoo" else get_alpha(ticker,start,end)

    # berita
    df_news, df_daily, wc, common, media = get_news(kw_encoded,start,end)

    if df_s is None or df_daily is None:
        st.error("Data gagal diambil")
        st.stop()

    df_merge = pd.merge(df_s, df_daily, on="Date", how="left")

    # ================= TABLE =================
    cols = ['Date','Prev_Close','Close']
    if not is_indo: cols.append('Close_IDR')
    cols += ['Price_Change','Pct_Change (%)','Sentiment','title']

    styled = df_merge[cols].style.applymap(color, subset=['Price_Change','Pct_Change (%)'])

    format_dict = {
        "Prev_Close": "{:.2f}",
        "Close": "{:.2f}",
        "Price_Change": "{:.2f}",
        "Pct_Change (%)": "{:.2f}"
    }

    if "Close_IDR" in df_merge.columns:
        format_dict["Close_IDR"] = "Rp {:,.2f}"

    st.dataframe(styled.format(format_dict), use_container_width=True)

    # ================= GRAFIK =================
    st.subheader("📈 Harga")
    st.line_chart(df_merge.set_index("Date")["Close"])

    # ================= PREDIKSI =================
    st.subheader("🤖 Prediksi LSTM")
    df_pred = lstm_predict(df_s)

    if df_pred is not None:
        fig, ax = plt.subplots()
        ax.plot(df_pred['Date'], df_pred['Close'], label="Actual")
        ax.plot(df_pred['Date'], df_pred['Prediksi'], label="Prediksi")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Data tidak cukup untuk prediksi")

    # ================= WORDCLOUD =================
    st.subheader("☁️ WordCloud")
    if wc:
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

    # ================= TOP WORD =================
    st.subheader("📊 Top Kata")
    if common:
        st.table(pd.DataFrame(common, columns=["Kata","Jumlah"]))

    # ================= MEDIA =================
    st.subheader("🏆 Top Media")
    if not media.empty:
        st.table(media.reset_index().rename(columns={"index":"Media"}))

    # ================= SENTIMENT =================
    st.subheader("📊 Distribusi Sentiment")
    st.bar_chart(df_news['sentiment_label'].value_counts())

    # ================= NEWS =================
    st.subheader("📰 Data Berita")
    st.dataframe(df_news)
