Here’s an updated version of your **README** incorporating the stock correlation analysis results while maintaining the existing structure:

---

# **Exploratory Data Analysis (EDA) on Stock News Headlines**  

This project combines comprehensive **Exploratory Data Analysis (EDA)** on a dataset containing over 1 million news headlines with **correlation analysis** to evaluate the relationship between news sentiment and stock price movements. The aim is to uncover trends, patterns, and actionable insights relevant to publication activity, sentiment, and stock behavior.  

---

## **Table of Contents**  
1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Methodology](#methodology)  
4. [Results and Insights](#results-and-insights)  
5. [Correlation Analysis](#correlation-analysis)  
6. [Challenges](#challenges)  
7. [Technologies Used](#technologies-used)  
8. [How to Use](#how-to-use)  

---

## **Project Overview**  
The dataset comprises **1,407,328 news headlines** from over 1,034 publishers, combined with stock price data for major companies (META, AMZN, TSLA, NVDA, GOOG, AAPL, MSFT). This project includes:  
- **Textual analysis** to extract headline lengths, sentiment, and topics.  
- **Time series analysis** to identify publishing patterns.  
- **Sentiment vs. Stock Returns** correlation analysis to evaluate how news sentiment affects daily stock price movements.  



## **Key Features**  
- **Descriptive Statistics:**  
   - Headline length analysis (mean: 73 characters, max: 512).  
   - Publisher activity (top contributor: Paul Quintaro with 228,373 articles).  

- **Text Analysis:**  
   - Topic modeling identified topics such as earnings reports, market reactions, and price targets.  
   - Sentiment analysis to classify articles as **positive**, **neutral**, or **negative**.  

- **Stock Correlation Analysis:**  
   - Analyzed relationships between **daily sentiment scores** and **stock price returns** for seven major stocks.  

- **Time Series Analysis:**  
   - Publication trends by day, month, and hour.  



## **Methodology**  
1. **Data Preprocessing:**  
   - Handled missing values and ensured consistency between news dates and stock trading days.  

2. **Sentiment Analysis:**  
   - News headlines were analyzed to compute daily average sentiment scores using NLP techniques.  

3. **Stock Movement Analysis:**  
   - Calculated daily percentage stock price returns based on adjusted closing prices.  

4. **Correlation Analysis:**  
   - Performed Pearson correlation between sentiment scores and stock returns to assess the strength and direction of the relationship.  

5. **Descriptive Statistics & NLP:**  
   - Conducted headline length analysis, publisher contributions, and topic modeling.  



## **Results and Insights**  

### **Descriptive Statistics**  
- **Headline Lengths:** Average length is 73 characters, with 50% of headlines between 47 and 87 characters.  
- **Publisher Contributions:**  
   - Top contributors include **Paul Quintaro**, **Lisa Levin**, and **Benzinga Newsdesk**.  

### **Text Analysis**  
- **Top Topics Identified:**  
   - **Topic 1:** Earnings reports and estimates (e.g., "EPS", "sales", "Q4").  
   - **Topic 2:** Price targets and ratings (e.g., "upgrade", "downgrade", "target").  
   - **Topic 3:** Market reactions (e.g., "shares", "trading", "performance").  

### **Time Series Analysis**  
- **Daily Trends:** Peak publication activity occurs on Thursdays; weekends show minimal activity.  
- **Hourly Trends:** Most news articles are published during market hours, with a sharp peak at **10:00 AM**.  



## **Correlation Analysis**  

The correlation analysis examined the relationship between **average daily sentiment scores** and **daily stock returns**. Results for each stock are summarized below:

| **Stock**                | **Correlation** | **P-Value** | **Observation**                               |
|--------------------------|-----------------|------------|---------------------------------------------|
| META                     | -0.0061         | 0.7943     | Weak negative correlation, not significant.  |
| AMZN                     | -0.0194         | 0.3592     | Weak negative correlation, not significant.  |
| TSLA                     | 0.0277          | 0.1909     | Weak positive correlation, not significant.  |
| NVDA                     | 0.0091          | 0.6668     | Weak positive correlation, not significant.  |
| GOOG                     | 0.0143          | 0.5007     | Weak positive correlation, not significant.  |
| AAPL                     | -0.0028         | 0.8944     | Negligible negative correlation.             |
| MSFT                     | -0.0118         | 0.5776     | Weak negative correlation, not significant.  |

**Key Observations:**  
- All correlation coefficients are close to **zero**, indicating negligible relationships between news sentiment and stock price movements.  
- High **p-values** (greater than 0.05) suggest that the correlations are not statistically significant.  



## **Challenges**  
- **Date Alignment:** Aligning sentiment data with valid stock trading days required rigorous data normalization.  
- **Sparse Sentiment Data:** On certain days, a limited number of news articles resulted in less representative sentiment scores.  
- **Low Correlation:** Results suggest external factors (e.g., market forces) may dominate short-term stock price changes, reducing sentiment influence.  



## **Technologies Used**  
- **Python**: Primary language for data processing and analysis.  
- **Libraries:**  
   - `pandas`, `numpy`: Data manipulation and statistical analysis.  
   - `matplotlib`, `seaborn`: Visualizations.  
   - `nltk`, `scikit-learn`: Sentiment analysis and NLP tools.  
   - `TA-Lib`, `PyNance`: Stock price indicators and financial metrics.  



## **How to Use**  

1. **Clone this repository:**  
   ```bash  
   git clone https://github.com/Azazh/Sentiment-Driven-Financial-Insights.git  
   

2. **Install the required libraries:**  
   ```bash  
   pip install -r requirements.txt  
   

3. **Run the analysis notebook:**  
   - Open and execute `Stock_Analysis_Insights.ipynb` in Jupyter Notebook.  

4. **Output Files:**  
   - Analysis results and visualizations will be generated under the `output/` directory.  


