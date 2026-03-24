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

# ================= CONFIG =================
HEADERS = {"User-Agent": "Mozilla/5.0"}

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# ================= SENTIMENT LEXICON =================
positive_words = {"naik","untung","laba","positif","menguat","tumbuh","baik","optimis"}
negative_words = {"turun","rugi","negatif","melemah","anjlok","buruk","krisis"}

def sentiment_score(text):
    words = text.split()
    score = 0
    for w in words:
        if w in positive_words: score += 1
        if w in negative_words: score -= 1
    return score

def sentiment_label(score):
    if score > 0: return "Positif"
    if score < 0: return "Negatif"
    return "Netral"

# ================= PREPROCESS =================
def preprocess(text):
    if pd.isna(text): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return stopword.remove(text)

# ================= YAHOO =================
def get_data(symbol, start, end):

    def to_unix(d):
        return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.JK"

    r = requests.get(url, headers=HEADERS, params={
        "interval": "1d",
        "period1": to_unix(start),
        "period2": to_unix(end + timedelta(days=1))
    })

    data = r.json()
    result = data.get("chart", {}).get("result")

    if not result:
        return None

    ts = result[0]["timestamp"]
    close = result[0]["indicators"]["quote"][0]["close"]

    df = pd.DataFrame({
        "Date":[datetime.fromtimestamp(t).date() for t in ts],
        "Close":close
    }).dropna()

    df['Prev_Close'] = df['Close'].shift(1)
    df['Change'] = df['Close'] - df['Prev_Close']

    return df

# ================= NEWS =================
def get_news(keyword, start, end):

    keyword = urllib.parse.quote(keyword)
    data = []

    for d in pd.date_range(start, end):
        url = f"https://news.google.com/rss/search?q={keyword}+after:{d.date()}+before:{(d+timedelta(days=1)).date()}&hl=id&gl=ID&ceid=ID:id"
        feed = feedparser.parse(url)

        for e in feed.entries:
            data.append({
                "Date": pd.to_datetime(e.published).date(),
                "title": e.title
            })

    df = pd.DataFrame(data).drop_duplicates()

    df["clean"] = df["title"].apply(preprocess)
    df["score"] = df["clean"].apply(sentiment_score)
    df["sentiment"] = df["score"].apply(sentiment_label)

    # agregasi harian
    df_daily = df.groupby("Date")["score"].mean().reset_index()
    df_daily.rename(columns={"score":"Sentiment"}, inplace=True)

    return df, df_daily

# ================= LSTM =================
def lstm_predict(df):

    data = df[['Close']].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(10, len(data_scaled)):
        X.append(data_scaled[i-10:i])
        y.append(data_scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X.shape[1],1)),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, verbose=0)

    pred = model.predict(X)
    pred = scaler.inverse_transform(pred)

    df_pred = df.iloc[10:].copy()
    df_pred['Prediksi'] = pred

    return df_pred

# ================= UI =================
st.title("🔥 AI Saham + Sentiment + Prediksi")

ticker = st.text_input("Ticker IDX", "BBCA")
keyword = st.text_input("Keyword Berita", "Bank BCA")

start = st.date_input("Start", datetime.now()-timedelta(days=60))
end = st.date_input("End", datetime.now())

if st.button("RUN AI"):

    df_s = get_data(ticker, start, end)
    df_news, df_sent = get_news(keyword, start, end)

    df_merge = pd.merge(df_s, df_sent, on="Date", how="left")

    # ================= PREDIKSI =================
    df_pred = lstm_predict(df_s)

    # ================= DISPLAY =================
    st.subheader("📊 Data Gabungan")
    st.dataframe(df_merge)

    st.subheader("📈 Grafik Harga + Prediksi")
    fig, ax = plt.subplots()
    ax.plot(df_pred['Date'], df_pred['Close'], label="Actual")
    ax.plot(df_pred['Date'], df_pred['Prediksi'], label="Prediksi")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📰 Sentiment Berita")
    st.dataframe(df_news)

    st.subheader("📊 Distribusi Sentiment")
    st.bar_chart(df_news['sentiment'].value_counts())
