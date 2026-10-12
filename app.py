import streamlit as st
import feedparser
import pandas as pd
from transformers import pipeline
import urllib.parse

# 1. Load the AI Model (Cached so it only loads once per server restart)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

# 2. Fetch Live News via Google News RSS (Restricted to Premium Sources & Last 7 Days)
def get_recent_news(ticker, max_results=15):
    trusted_sources = "site:moneycontrol.com OR site:economictimes.indiatimes.com OR site:livemint.com OR site:business-standard.com"
    time_filter = "when:7d"
    search_query = f'"{ticker}" {trusted_sources} {time_filter}'
    
    query = urllib.parse.quote_plus(search_query)
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    feed = feedparser.parse(rss_url)
    headlines = []
    
    for entry in feed.entries[:max_results]:
        headlines.append({
            "Title": entry.title,
            "Published": entry.published,
            "Link": entry.link
        })
        
    return pd.DataFrame(headlines)

# 3. Score the Headlines
def analyze_sentiment(df, model):
    scores = []
    labels = []
    score_mapping = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    
    for title in df['Title']:
        result = model(title)[0]
        label = result['label'].lower()
        
        labels.append(label.capitalize())
        scores.append(score_mapping[label] * result['score'])
        
    df['Sentiment'] = labels
    df['Score Value'] = scores
    return df

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Market Sentiment Scanner", layout="wide")
st.title("Live Stock Sentiment Analyzer 📈")
st.markdown("Instantly score the latest premium Indian news (past 7 days) using FinBERT.")

model = load_sentiment_model()

col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input("Enter NSE/BSE Ticker (e.g., RELIANCE, TCS, INFY):")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("Analyze Sentiment", use_container_width=True)

if analyze_btn and ticker_input:
    with st.spinner(f"Fetching live news and analyzing sentiment for {ticker_input.upper()}..."):
        news_df = get_recent_news(ticker_input.upper())
        
        if news_df.empty:
            st.warning("No recent news found for this ticker from trusted sources.")
        else:
            scored_df = analyze_sentiment(news_df, model)
            
            # Filter out neutral headlines to reveal true directional sentiment
            directional_scores = scored_df[scored_df['Sentiment'] != 'Neutral']['Score Value']
            
            if directional_scores.empty:
                raw_score = 0.0
            else:
                raw_score = directional_scores.mean()
            
            # Convert raw score (-1.0 to +1.0) into a 1 to 10 scale
            scaled_score = round(((raw_score + 1.0) / 2.0) * 9.0 + 1.0, 1)
            
            st.subheader("Current Market Sentiment")
            
            # 1 to 10 scale classifications
            if scaled_score >= 8.5:
                sentiment_text, color = "Strong Bullish 🟢", "green"
            elif scaled_score >= 6.5:
                sentiment_text, color = "Mildly Bullish 📈", "lightgreen"
            elif scaled_score >= 4.5:
                sentiment_text, color = "Neutral ⚪", "gray"
            elif scaled_score >= 2.5:
                sentiment_text, color = "Mildly Bearish 📉", "orange"
            else:
                sentiment_text, color = "Strong Bearish 🔴", "red"
                
            st.markdown(
                f"<h2 style='color: {color};'>{sentiment_text} — {scaled_score}/10</h2>", 
                unsafe_allow_html=True
            )
            
            st.subheader(f"Recent Headlines ({len(scored_df)} articles analyzed)")
            display_df = scored_df[['Published', 'Title', 'Sentiment', 'Score Value']]
            st.dataframe(display_df, use_container_width=True)
