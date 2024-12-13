import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class EDAAnalysis:
    def __init__(self, file_path):
        """
        Initialize the class with the dataset path.
        """
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """
        Load the dataset from the file path.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print("Data loaded successfully!")
        except Exception as e:
            print(f"Error loading data: {e}")

    def calculate_headline_length(self):
        """
        Calculate and analyze headline lengths.
        """
        if self.data is None:
            raise ValueError("Data is not loaded. Call load_data() first.")
        self.data['headline_length'] = self.data['headline'].apply(len)
        print(self.data['headline_length'].describe())

        # Plot the distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['headline_length'], bins=30, kde=True, color='blue')
        plt.title('Distribution of Headline Lengths')
        plt.xlabel('Headline Length')
        plt.ylabel('Frequency')
        plt.show()

    def count_articles_by_publisher(self):
        """
        Count and visualize the number of articles per publisher.
        """
        if self.data is None:
            raise ValueError("Data is not loaded. Call load_data() first.")
        publisher_counts = self.data['publisher'].value_counts()
        print(publisher_counts)

        # Plot top publishers
        plt.figure(figsize=(12, 6))
        publisher_counts.head(10).plot(kind='bar', color='green')
        plt.title('Top 10 Publishers by Article Count')
        plt.xlabel('Publisher')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.show()

    def analyze_publication_dates(self):
        """
        Analyze the publication dates to identify trends over time,
        such as increased news frequency on particular days or events.
        """
        if self.data is None:
            raise ValueError("Data is not loaded. Call load_data() first.")
        
        # Handle datetime parsing with flexible format and coercion for invalid values
        self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
        
        # Drop rows with invalid or missing dates
        self.data = self.data.dropna(subset=['date'])

        # Extract day of the week and month
        self.data['day_of_week'] = self.data['date'].dt.day_name()
        self.data['month'] = self.data['date'].dt.month_name()

        # Count articles by day of the week
        articles_by_day = self.data['day_of_week'].value_counts()
        print("Articles by day of the week:")
        print(articles_by_day)

        # Count articles by month
        articles_by_month = self.data['month'].value_counts()
        print("\nArticles by month:")
        print(articles_by_month)
    def perform_sentiment_analysis(self):
        """
        Perform sentiment analysis on the headlines.
        Adds a 'sentiment' column to the DataFrame indicating sentiment polarity.
        """
        if self.data is None:
            raise ValueError("Data is not loaded. Call load_data() first.")
        
        if 'headline' not in self.data.columns:
            raise ValueError("The dataset must contain a 'headline' column.")
        
        def get_sentiment(text):
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            if polarity > 0:
                return 'positive'
            elif polarity < 0:
                return 'negative'
            else:
                return 'neutral'

        self.data['sentiment'] = self.data['headline'].apply(get_sentiment)
        print("Sentiment analysis completed.")
        print(self.data['sentiment'].value_counts())

    def perform_topic_modeling(self, num_topics=5, num_keywords=10):
        """
        Perform topic modeling on the headlines.
        Uses LDA (Latent Dirichlet Allocation) to extract topics and keywords.
        """
        if self.data is None:
            raise ValueError("Data is not loaded. Call load_data() first.")

        if 'headline' not in self.data.columns:
            raise ValueError("The dataset must contain a 'headline' column.")

        vectorizer = CountVectorizer(stop_words='english')
        dtm = vectorizer.fit_transform(self.data['headline'])
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(dtm)

        # Extract topics and keywords
        topics = {}
        for i, topic in enumerate(lda.components_):
            keywords = [vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-num_keywords:]]
            topics[f"Topic {i + 1}"] = keywords
        
        print("Topic modeling completed.")
        for topic, keywords in topics.items():
            print(f"{topic}: {', '.join(keywords)}")