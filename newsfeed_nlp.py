import streamlit as st
import feedparser
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import nltk

# Download the VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to fetch and parse RSS feed
def fetch_rss_feed(ticker):
    # URL format for Yahoo Finance RSS feed
    feed_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    feed = feedparser.parse(feed_url)
    return feed

# Function to perform sentiment analysis using VADER
def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound_score = scores['compound']
    return compound_score

# Function to generate a word cloud from text
def generate_wordcloud(text):
    wordcloud = WordCloud(
        background_color='white',
        width=800,
        height=400,
        max_words=100,
        colormap='viridis'
    ).generate(text)
    
    st.sidebar.subheader("Word Cloud of Yahoo News Headlines")
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.sidebar.pyplot(plt)

# Streamlit app
def main():
    st.title("Stock News Sentiment Analysis")

    # User input for ticker symbol
    ticker = st.text_input("Enter a ticker symbol (e.g., GOOGL, AAPL, MSFT):", value="GOOGL").upper()

    if ticker:
        with st.spinner("Fetching news..."):
            feed = fetch_rss_feed(ticker)
            
            if feed.entries:
                # Collect headlines for word cloud generation and separate lists for sentiment categories
                all_headlines = ""
                positive_news = []
                neutral_news = []
                negative_news = []

                # Loop through each news entry and perform sentiment analysis
                for entry in feed.entries:
                    title = entry.title
                    link = entry.link
                    published = entry.published
                    
                    # Concatenate titles for the word cloud
                    all_headlines += f"{title} "
                    
                    # Analyze the sentiment of the news title
                    sentiment_score = analyze_sentiment(title)
                    
                    # Classify news based on sentiment score
                    if sentiment_score > 0.05:
                        sentiment = "Positive"
                        positive_news.append({
                            "Title": title,
                            "Score": round(sentiment_score, 2),
                            "Published": published,
                            "Link": link
                        })
                    elif sentiment_score < -0.05:
                        sentiment = "Negative"
                        negative_news.append({
                            "Title": title,
                            "Score": round(sentiment_score, 2),
                            "Published": published,
                            "Link": link
                        })
                    else:
                        sentiment = "Neutral"
                        neutral_news.append({
                            "Title": title,
                            "Score": round(sentiment_score, 2),
                            "Published": published,
                            "Link": link
                        })
                
                # Generate and display the word cloud in the sidebar if there are headlines
                if all_headlines:
                    generate_wordcloud(all_headlines)

                # Display sentiment summary in the sidebar
                st.sidebar.subheader("Sentiment Summary")
                st.sidebar.write(f"**Total Positive News:** {len(positive_news)}")
                st.sidebar.write(f"**Total Neutral News:** {len(neutral_news)}")
                st.sidebar.write(f"**Total Negative News:** {len(negative_news)}")

                # Display news entries on the main page
                st.subheader(f"News Sentiment Analysis for {ticker}")

                # Positive news section
                if positive_news:
                    st.markdown("### Positive News")
                    for news in positive_news:
                        st.write(f"**Title:** {news['Title']}")
                        st.write(f"**Score:** {news['Score']}")
                        st.write(f"**Published:** {news['Published']}")
                        st.write(f"**Link:** [Read more]({news['Link']})")
                        st.write("---")
                
                # Neutral news section
                if neutral_news:
                    st.markdown("### Neutral News")
                    for news in neutral_news:
                        st.write(f"**Title:** {news['Title']}")
                        st.write(f"**Score:** {news['Score']}")
                        st.write(f"**Published:** {news['Published']}")
                        st.write(f"**Link:** [Read more]({news['Link']})")
                        st.write("---")
                
                # Negative news section
                if negative_news:
                    st.markdown("### Negative News")
                    for news in negative_news:
                        st.write(f"**Title:** {news['Title']}")
                        st.write(f"**Score:** {news['Score']}")
                        st.write(f"**Published:** {news['Published']}")
                        st.write(f"**Link:** [Read more]({news['Link']})")
                        st.write("---")
                
            else:
                st.write("No news found for the given ticker symbol.")

if __name__ == "__main__":
    main()
