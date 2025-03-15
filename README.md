## Project Description
This project is a keyword-based news crawling and analysis tool built using Google News, Streamlit, and NLP techniques. It allows users to search news articles based on a given keyword, preprocess the collected data, and analyze the most frequently occurring words. The project features word cloud visualization, top word frequency analysis, and media source ranking, providing insights into trending topics in real-time.

# 🔍 Keyword News Crawling & Analysis  

## 📌 Overview  
This project is a **news crawling and analysis tool** that collects articles from **Google News** based on a keyword, processes the text using NLP techniques, and provides visual insights using **word clouds and statistical analysis**. It is built with **Streamlit** for an interactive UI and **GoogleNews API** for data collection.  

## 🛠 Features  
✅ **Real-time news crawling** using Google News API  
✅ **Text preprocessing** with stopword removal  
✅ **Word cloud visualization** for key insights  
✅ **Top word frequency analysis**  
✅ **Media source ranking**  
✅ **Interactive UI using Streamlit**  

## 🖥️ Installation  
### Prerequisites  
- Python 3.8+  
- GoogleNews  
- Streamlit  
- Sastrawi (for Indonesian stopword removal)  
- Matplotlib  
- Pandas  

🔍 Usage
Enter a keyword (e.g., "Teknologi", "Ekonomi").
The tool will crawl Google News and collect articles.
Process the word cloud and word frequency analysis.
View top media sources reporting on the topic.
📂 Project Structure
📁 news-crawling-analysis  
│── 📁 scripts/               # Processing scripts  
│── app.py                    # Main Streamlit application  
│── requirements.txt           # Dependencies  
│── README.md                  # Project documentation  

### Setup  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/news-crawling-analysis.git
   cd news-crawling-analysis
