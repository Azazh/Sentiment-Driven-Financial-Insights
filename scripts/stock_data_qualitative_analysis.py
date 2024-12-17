import pandas as pd
from scipy.stats import pearsonr
import talib
import matplotlib.pyplot as plt
import os


class StockDataAnalyzer:
    def __init__(self, data_path: str):
        """
        Initialize the StockDataAnalyzer with the directory containing stock data.

        Args:
            data_path (str): Path to the directory containing stock CSV files.
        """
        self.data_path = data_path
        self.dataframes = {}  # Dictionary to store dataframes for each stock.

    def load_data(self, file_name: str):
        """
        Load a single stock data CSV file into a pandas DataFrame.

        Args:
            file_name (str): CSV file name.

        Returns:
            pd.DataFrame: DataFrame containing the stock data.
        """
        file_path = os.path.join(self.data_path, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            self.dataframes[file_name] = df
            return df
        else:
            raise FileNotFoundError(f"{file_name} not found in {self.data_path}")

    def calculate_technical_indicators(self, df: pd.DataFrame):
        """
        Calculate technical indicators using TA-Lib and add them to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing stock data.

        Returns:
            pd.DataFrame: DataFrame with additional columns for indicators.
        """
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        return df

    def visualize_data(self, df: pd.DataFrame, stock_name: str):
        """
        Create visualizations for stock data and indicators.

        Args:
            df (pd.DataFrame): DataFrame containing stock data.
            stock_name (str): Name of the stock for visualization title.
        """
        plt.figure(figsize=(14, 8))

        # Plot Close price and SMAs
        plt.subplot(3, 1, 1)
        plt.plot(df['Close'], label='Close Price', color='blue')
        plt.plot(df['SMA_50'], label='SMA 50', color='orange')
        plt.plot(df['SMA_200'], label='SMA 200', color='green')
        plt.title(f"{stock_name} - Close Price and Moving Averages")
        plt.legend()

        # Plot RSI
        plt.subplot(3, 1, 2)
        plt.plot(df['RSI'], label='RSI', color='purple')
        plt.axhline(70, color='red', linestyle='--', linewidth=0.8)
        plt.axhline(30, color='green', linestyle='--', linewidth=0.8)
        plt.title("RSI")
        plt.legend()

        # Plot MACD
        plt.subplot(3, 1, 3)
        plt.plot(df['MACD'], label='MACD', color='blue')
        plt.plot(df['MACD_Signal'], label='MACD Signal', color='red')
        plt.bar(df.index, df['MACD'] - df['MACD_Signal'], label='MACD Histogram', color='gray')
        plt.title("MACD")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def analyze_all_stocks(self):
        """
        Analyze all stock data files in the directory.
        Display correlation and p-value between Close price and RSI.
        """
        for file_name in os.listdir(self.data_path):
            if file_name.endswith('.csv'):
                print(f"Analyzing {file_name}...")
                df = self.load_data(file_name)
                df = self.calculate_technical_indicators(df)
                
                # Drop NaN values to ensure valid correlation calculations
                valid_data = df[['Close', 'RSI']].dropna()
                
                # Calculate correlation and p-value
                correlation, p_value = pearsonr(valid_data['Close'], valid_data['RSI'])
                print(f"{file_name}: Correlation={correlation:.4f}, P-Value={p_value:.4f}")
                
                # Call the visualization
                self.visualize_data(df, stock_name=file_name.split('_')[0])
