import pandas as pd
from textblob import TextBlob
from datetime import datetime
from scipy.stats import pearsonr
import os
import matplotlib.pyplot as plt


class StockSentimentAnalyzer:
    def __init__(self, stock_data_path, news_data_path):
        """
        Initialize paths to stock and news datasets.
        """
        self.stock_data_path = stock_data_path
        self.news_data_path = news_data_path
    
    @staticmethod
    def normalize_dates(df, date_column, date_format=None):
        """
        Normalize dates to a standard format.
        """
        df[date_column] = pd.to_datetime(df[date_column], format=date_format, errors='coerce')
        return df
    
    @staticmethod
    def calculate_sentiment(headline):
        """
        Calculate sentiment polarity using TextBlob.
        """
        return TextBlob(headline).sentiment.polarity
    
    def load_stock_data(self, file_path):
        """
        Load and prepare stock data.
        """
        stock_df = pd.read_csv(file_path)
        stock_df = self.normalize_dates(stock_df, 'Date')
        stock_df.sort_values(by='Date', inplace=True)
        stock_df['Daily_Return'] = stock_df['Adj Close'].pct_change()
        return stock_df
    
    def load_and_process_news(self):
        """
        Load news data and calculate sentiment scores.
        """
        news_df = pd.read_csv(self.news_data_path)
        news_df = self.normalize_dates(news_df, 'date')
        news_df['Sentiment'] = news_df['headline'].apply(self.calculate_sentiment)
        return news_df
    
    def aggregate_sentiments(self, news_df):
        """
        Aggregate sentiment scores by date.
        """
        aggregated_news = news_df.groupby(news_df['date'].dt.date)['Sentiment'].mean().reset_index()
        aggregated_news.rename(columns={'date': 'Date', 'Sentiment': 'Avg_Sentiment'}, inplace=True)
        
        # Ensure 'Date' is in datetime64 format
        aggregated_news['Date'] = pd.to_datetime(aggregated_news['Date'])
        return aggregated_news

    
    def correlate_sentiment_with_stock(self, stock_file):
        """
        Correlate average sentiment scores with stock daily returns.
        """
        stock_df = self.load_stock_data(stock_file)
        news_df = self.load_and_process_news()
        aggregated_news = self.aggregate_sentiments(news_df)
        
        # Merge stock and news data on Date
        merged_df = pd.merge(stock_df, aggregated_news, on='Date', how='inner')
        merged_df = merged_df.dropna(subset=['Daily_Return', 'Avg_Sentiment'])
        
        # Calculate Pearson Correlation
        correlation, p_value = pearsonr(merged_df['Avg_Sentiment'], merged_df['Daily_Return'])
        return correlation, p_value, merged_df

    def analyze_all_stocks(self):
        """
        Analyze sentiment correlation for all stock files in a directory.
        """
        results = {}
        for file in os.listdir(self.stock_data_path):
            if file.endswith(".csv"):
                print(f"Analyzing {file}...")
                file_path = os.path.join(self.stock_data_path, file)
                correlation, p_value, _ = self.correlate_sentiment_with_stock(file_path)
                results[file] = {'Correlation': correlation, 'P-Value': p_value}
        
        return results

    def visualize_all_stocks(self):
        """
        Visualize sentiment and daily returns for all stock files.
        """
        for file in os.listdir(self.stock_data_path):
            if file.endswith(".csv"):
                print(f"Visualizing {file}...")
                file_path = os.path.join(self.stock_data_path, file)
                
                try:
                    _, _, merged_df = self.correlate_sentiment_with_stock(file_path)
                    
                    # Plot Daily Returns and Sentiment
                    plt.figure(figsize=(12, 6))
                    plt.title(f"Sentiment vs Daily Returns: {file}")
                    plt.plot(merged_df['Date'], merged_df['Daily_Return'], label='Daily Return', color='blue')
                    plt.plot(merged_df['Date'], merged_df['Avg_Sentiment'], label='Average Sentiment', color='orange')
                    plt.xlabel("Date")
                    plt.ylabel("Values")
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
                    
                except Exception as e:
                    print(f"Failed to process {file}. Error: {e}")

