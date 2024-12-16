import re
from typing import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud


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
        # Extract unique email domains if present
        email_publishers = self.data['publisher'].str.contains('@', na=False)
        unique_domains = (
            self.data.loc[email_publishers, 'publisher']
            .str.split('@').str[1]
            .value_counts()
        )
        if not unique_domains.empty:
            print("\nUnique Domains:")
            print(unique_domains)
        else:
            print("\nNo email addresses found among publishers.")

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

        # Plotting the results
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot articles by day of the week
        articles_by_day.plot(kind='bar', ax=axes[0], color='skyblue')
        axes[0].set_title('Articles by Day of the Week')
        axes[0].set_xlabel('Day of the Week')
        axes[0].set_ylabel('Number of Articles')

        # Plot articles by month
        articles_by_month.plot(kind='bar', ax=axes[1], color='lightgreen')
        axes[1].set_title('Articles by Month')
        axes[1].set_xlabel('Month')
        axes[1].set_ylabel('Number of Articles')

        plt.tight_layout()
        plt.show()

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

        # Apply sentiment analysis
        self.data['sentiment'] = self.data['headline'].apply(get_sentiment)
        print("Sentiment analysis completed.")
        print(self.data['sentiment'].value_counts())

        # Plot sentiment distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(x='sentiment', data=self.data, palette='Set2')
        plt.title('Sentiment Distribution of Headlines')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.show()

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

        # Plot the topic weights (importance of each topic)
        plt.figure(figsize=(8, 6))
        topic_weights = [sum(topic) for topic in lda.components_]
        plt.bar(range(1, num_topics + 1), topic_weights, color='skyblue')
        plt.title('Topic Weights Distribution')
        plt.xlabel('Topic')
        plt.ylabel('Weight')
        plt.xticks(range(1, num_topics + 1))
        plt.show()

        # WordCloud for each topic
        for i, topic in enumerate(lda.components_):
            plt.figure(figsize=(8, 6))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
                {vectorizer.get_feature_names_out()[index]: topic[index] for index in topic.argsort()[-num_keywords:]})
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"Word Cloud for Topic {i + 1}")
            plt.show()

    def preprocess_data(self):
        """
        Preprocess the dataset to ensure the date column is in datetime format.
        """
        if self.data is None:
            raise ValueError("Data is not loaded. Call load_data() first.")

        # Ensure the 'date' column is in datetime format
        self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')

        # Drop rows with invalid or missing dates
        self.data.dropna(subset=['date'], inplace=True)
        print("Data preprocessing completed.")

    def analyze_publication_frequency(self):
        """
        Analyze publication frequency over time.
        """
        if self.data is None:
            raise ValueError("Data is not loaded. Call load_data() first.")

        # Group by date and count the number of articles per day
        daily_publications = self.data.groupby(self.data['date'].dt.date).size()

        # Print the publication frequency for inspection
        print(daily_publications)

        # Plot the publication frequency over time
        plt.figure(figsize=(12, 6))
        daily_publications.plot(kind='line', color='blue')
        plt.title('Publication Frequency Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.grid(True)
        plt.show()

    def analyze_publishing_times(self):
        """
        Analyze the times of the day when most articles are published.
        """
        if self.data is None:
            raise ValueError("Data is not loaded. Call load_data() first.")

        # Extract the hour of publication
        self.data['hour'] = self.data['date'].dt.hour

        # Group by hour and count the number of articles
        hourly_publications = self.data.groupby('hour').size()

        # Print the hourly publication counts for inspection
        print(hourly_publications)

        # Plot the distribution of publishing times
        plt.figure(figsize=(12, 6))
        hourly_publications.plot(kind='bar', color='green')
        plt.title('Distribution of Publishing Times')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Articles')
        plt.grid(True)
        plt.show()

    def extract_keywords(self):
        """
        Extract keywords from the URL column for analysis.
        """
        if self.data is None:
            print("No data loaded. Please load the data first.")
            return None

        def extract(url):
            keywords = re.findall(r'\b\w+\b', url)  # Extract words
            return [kw.lower() for kw in keywords if kw.isalpha()]  # Filter alphabetic words

        self.data['keywords'] = self.data['url'].apply(extract)

    print("Keywords extracted successfully.")

    def categorize_news(self):
        """
        Categorize the news into predefined categories based on keywords.
        """
        if 'keywords' not in self.data.columns:
            print("Keywords not found. Please extract keywords first.")
            return None

        # Define categories and their associated keywords
        categories = {
            'finance': ['stock', 'market', 'investment', 'profit', 'earnings'],
            'technology': ['tech', 'software', 'ai', 'innovation', 'hardware'],
            'healthcare': ['health', 'medicine', 'pharma', 'vaccine', 'doctor'],
            'energy': ['oil', 'gas', 'energy', 'renewable', 'solar'],
        }

        def map_category(keywords):
            matched_categories = []
            for category, words in categories.items():
                if any(word in keywords for word in words):
                    matched_categories.append(category)
            return matched_categories if matched_categories else ['other']

        self.data['categories'] = self.data['keywords'].apply(map_category)
        print("News categorized successfully.")

    def analyze_category_by_publisher(self):
        """
        Analyze the distribution of news categories for each publisher.
        """
        if 'categories' not in self.data.columns:
            print("Categories not found. Please categorize news first.")
            return None

        # Flatten the category lists for analysis
        self.data['categories_flat'] = self.data['categories'].apply(lambda x: ','.join(x))

        category_distribution = self.data.groupby('publisher')['categories_flat'].apply(
            lambda x: Counter(','.join(x).split(',')).most_common())
        print("Category analysis by publisher completed.")
        return category_distribution

    def plot_publisher_contributions(self, top_n=10):
        """
        Plot the top publishers contributing to the news feed.
        """
        if self.data is None:
            print("No data loaded. Please load the data first.")
            return None

        publisher_counts = self.data['publisher'].value_counts().head(top_n)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=publisher_counts.values, y=publisher_counts.index, palette="viridis")
        plt.title(f"Top {top_n} Publishers Contributing to the News Feed")
        plt.xlabel("Number of Articles")
        plt.ylabel("Publisher")
        plt.show()

    def plot_category_distribution(self):
        """
        Plot the distribution of news categories.
        """
        if 'categories' not in self.data.columns:
            print("Categories not found. Please categorize news first.")
            return None

        # Flatten and count category frequencies
        all_categories = [cat for sublist in self.data['categories'] for cat in sublist]
        category_counts = Counter(all_categories)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(category_counts.values()), y=list(category_counts.keys()), palette="viridis")
        plt.title("Distribution of News Categories")
        plt.xlabel("Number of Articles")
        plt.ylabel("Category")
        plt.show()

    def extract_domains_from_publishers(self):
        """
        Extract unique domains from publishers if email-style names are used.
        """
        if self.data is None:
            print("No data loaded. Please load the data first.")
            return None

        def extract_domain(publisher):
            if "@" in publisher:
                domain = publisher.split("@")[-1]
                return domain
            return None

        self.data['domains'] = self.data['publisher'].apply(extract_domain)
        domain_counts = self.data['domains'].value_counts()
        print("Domain extraction completed.")
        return domain_counts

    def plot_top_domains(self, top_n=10):
        """
        Plot the top domains contributing to the news feed.
        """
        if 'domains' not in self.data.columns:
            print("Domains not found. Please extract domains first.")
            return None

        domain_counts = self.data['domains'].value_counts().head(top_n)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=domain_counts.values, y=domain_counts.index, palette="viridis")
        plt.title(f"Top {top_n} Domains Contributing to the News Feed")
        plt.xlabel("Number of Articles")
        plt.ylabel("Domain")
        plt.show()
