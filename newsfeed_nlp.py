import streamlit as st
import feedparser
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
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
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# Streamlit app
def main():
    st.title("Stock News Sentiment Analysis with Word Cloud")

    # User input for ticker symbol
    ticker = st.text_input("Enter a ticker symbol (e.g., GOOGL, AAPL, MSFT):", value="GOOGL").upper()

    if ticker:
        with st.spinner("Fetching news..."):
            feed = fetch_rss_feed(ticker)
            
            if feed.entries:
                st.subheader(f"Recent News for {ticker} with Sentiment Analysis:")
                
                # Collect headlines for word cloud generation
                all_headlines = ""

                # Loop through each news entry and perform sentiment analysis
                for entry in feed.entries:
                    title = entry.title
                    link = entry.link
                    published = entry.published
                    
                    # Concatenate titles for the word cloud
                    all_headlines += f"{title} "
                    
                    # Analyze the sentiment of the news title
                    sentiment_score = analyze_sentiment(title)
                    
                    # Determine sentiment label based on the score
                    if sentiment_score > 0.05:
                        sentiment = "Positive"
                    elif sentiment_score < -0.05:
                        sentiment = "Negative"
                    else:
                        sentiment = "Neutral"
                    
                    # Display the news entry with sentiment analysis
                    st.write(f"**Title:** {title}")
                    st.write(f"**Sentiment:** {sentiment} (Score: {sentiment_score:.2f})")
                    st.write(f"**Link:** [Read more]({link})")
                    st.write(f"**Published:** {published}")
                    st.write("---")
                
                # Generate and display the word cloud if there are headlines
                if all_headlines:
                    st.subheader("Word Cloud of News Headlines")
                    generate_wordcloud(all_headlines)
            else:
                st.write("No news found for the given ticker symbol.")

if __name__ == "__main__":
    main()
