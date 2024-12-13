# Exploratory Data Analysis (EDA) on Stock News Headlines  

This project involves performing comprehensive Exploratory Data Analysis (EDA) on a dataset containing over 1 million news headlines. The aim is to uncover trends, patterns, and actionable insights relevant to publication activity, headline content, and sentiment.  

---

## **Table of Contents**  
1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Methodology](#methodology)  
4. [Results and Insights](#results-and-insights)  
5. [Challenges](#challenges)  
6. [Technologies Used](#technologies-used)  
7. [How to Use](#how-to-use)  
8. [Future Improvements](#future-improvements)  

---

## **Project Overview**  
The dataset comprises 1,407,328 news headlines from over 1,034 publishers, with detailed analysis performed to explore:  
- Textual headline lengths.  
- Publication trends by publisher, day, and month.  
- Sentiment and topic modeling using Natural Language Processing (NLP).  
- Time series analysis to identify publishing patterns and spikes.  

---

## **Key Features**  
- **Descriptive Statistics:**  
  - Headline length analysis (mean: 73 characters, max: 512).  
  - Publisher activity (top contributor: Paul Quintaro with 228,373 articles).  

- **Text Analysis:**  
  - Topic modeling identified key topics like earnings, market reactions, and stock performance.  
  - Sentiment analysis to gauge article tone (positive, negative, or neutral).  

- **Time Series Analysis:**  
  - Publication peaks observed during market events.  
  - Articles are predominantly published between 7:00 AM and 11:00 AM, with a peak at 10:00 AM.  

---

## **Methodology**  
1. **Data Preprocessing:**  
   - Handled missing values and standardized formats for publication dates and publishers.  
2. **Descriptive Statistics:**  
   - Calculated headline lengths, publisher contributions, and domain counts.  
3. **Text Analysis:**  
   - Performed NLP-based topic modeling and sentiment analysis.  
4. **Time Series Analysis:**  
   - Analyzed trends by day, month, and publication hour.  


---

## **Results and Insights**  

### **Descriptive Statistics**  
- **Headline Lengths:** Average length is 73 characters, with 50% of headlines being between 47 and 87 characters.  
- **Publisher Contributions:** Dominated by a few publishers, with **Paul Quintaro**, **Lisa Levin**, and **Benzinga Newsdesk** being the top contributors.  

### **Text Analysis**  
- **Top Topics Identified:**  
  - **Topic 1:** Earnings reports and estimates (e.g., "EPS", "Q4", "sales").  
  - **Topic 2:** Price targets and ratings (e.g., "downgrade", "target", "price").  
  - **Topic 3:** Market reactions (e.g., "coronavirus", "trading", "shares").  

### **Time Series Analysis**  
- **Daily Trends:** Peak activity on Thursdays; minimal activity on weekends.  
- **Hourly Trends:** Most articles are published during trading hours, with a peak at **10:00 AM**.  

---

## **Challenges**  
- **Large Dataset:** Handling over a million entries required optimizing computations and storage.    

---

## **Technologies Used**  
- Python: For data analysis and NLP processing.  
- Libraries:  
  - `pandas`, `numpy`: Data manipulation and statistical analysis.  
  - `matplotlib`, `seaborn`: Data visualization.  
  - `nltk`, `scikit-learn`: Natural Language Processing and topic modeling.  

---

## **How to Use**  
1. Clone this repository:  
   ```bash  
   git clone https://github.com/Azazh/Sentiment-Driven-Financial-Insights.git  
2. Install the required libraries:
    pip install -r requirements.txt  
3. Run the analysis:
    Execute Stock_Analysis_Insights.ipynb
