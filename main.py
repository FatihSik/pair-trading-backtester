import yfinance as yf
import pandas as pd
import os
import datetime as dt
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

class Index:
    def __init__(self, name):
        self.name = name
        self.components = self.get_index_components()

    def get_index_components(self):
        if self.name == 'sandp':
            file = pd.read_csv('sandp.csv')
            return file['Components'].tolist()
        elif self.name == 'djia':
            file = pd.read_csv('djia.csv')
            return file['Components'].tolist()
        elif self.name == 'crypto':
            file = pd.read_csv('crypto.csv')
            return file['Components'].tolist()
        else:
            return []
    def length(self):
        return len(self.components)

class FetchData:
    def __init__(self, components, start, end):
        self.components = components
        self.start = start
        self.end = end
        self.directory = 'stock data'
        self.getData()

    def cleanData(self, df):
        df.drop([0, 1], axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.rename(columns={'Price': 'Date'}, inplace=True)

    def getData(self, start=None, end=None):
        start = start or self.start
        end = end or self.end

        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)
            for component in self.components:
                data = yf.download(component, start, end, progress=False)
                data.to_csv(f'{self.directory}/{component}.csv')
                df = pd.read_csv(f'{self.directory}/{component}.csv')
                if 'Unnamed: 0' in df.columns:
                    df.drop(columns=['Unnamed: 0'], inplace=True)
                self.cleanData(df)
                df.to_csv(f'{self.directory}/{component}.csv')
                print(f'{component} data downloaded')
        else:
            for component in self.components:
                ndl = pd.date_range(start=start, end=end).strftime('%Y-%m-%d').tolist()
                nmind = pd.to_datetime(min(ndl))
                nmaxd = pd.to_datetime(max(ndl))
                df = pd.read_csv(f'{self.directory}/{component}.csv')
                if 'Unnamed: 0' in df.columns:
                    df.drop(columns=['Unnamed: 0'], inplace=True)
                df['Date'] = pd.to_datetime(df['Date'])
                omind = df['Date'].min()
                omaxd = df['Date'].max()
                if nmind < omind and nmaxd > omaxd:
                    dataTop = yf.download(component, start=omaxd + dt.timedelta(days=1), end=nmaxd, progress=False)
                    dfTop = pd.DataFrame(dataTop)
                    dfTop.to_csv('top.csv')
                    dfTop = pd.read_csv('top.csv')
                    if 'Unnamed: 0' in dfTop.columns:
                        dfTop.drop(columns=['Unnamed: 0'], inplace=True)
                    self.cleanData(dfTop)
                    dataBottom = yf.download(component, start=nmind, end=omind - dt.timedelta(days=1), progress=False)
                    dfBottom = pd.DataFrame(dataBottom)
                    dfBottom.to_csv('bottom.csv')
                    dfBottom = pd.read_csv('bottom.csv')
                    if 'Unnamed: 0' in dfBottom.columns:
                        dfBottom.drop(columns=['Unnamed: 0'], inplace=True)
                    self.cleanData(dfBottom)
                    df = pd.concat([dfBottom, df, dfTop], axis=0)
                    df.reset_index(drop=True, inplace=True)
                    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                    df.drop_duplicates(subset=['Date'], keep='first', inplace=True)
                    df.to_csv(f'{self.directory}/{component}.csv')
                    print(f'{component} data updated')
                    os.remove('top.csv')
                    os.remove('bottom.csv')
                elif nmaxd > omaxd:
                    data = yf.download(component, start=omaxd + dt.timedelta(days=1), end=nmaxd, progress=False)
                    dfNew = pd.DataFrame(data)
                    dfNew.to_csv('new.csv')
                    dfNew = pd.read_csv('new.csv')
                    if 'Unnamed: 0' in dfNew.columns:
                        dfNew.drop(columns=['Unnamed: 0'], inplace=True)
                    self.cleanData(dfNew)
                    df = pd.concat([df, dfNew], axis=0)
                    df.reset_index(drop=True, inplace=True)
                    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                    df.drop_duplicates(subset=['Date'], keep='first', inplace=True)
                    df.to_csv(f'{self.directory}/{component}.csv')
                    print(f'{component} data updated')
                    os.remove('new.csv')
                elif nmind < omind:
                    data = yf.download(component, start=nmind, end=omind - dt.timedelta(days=1), progress=False)
                    dfNew = pd.DataFrame(data)
                    dfNew.to_csv('new.csv')
                    dfNew = pd.read_csv('new.csv')
                    if 'Unnamed: 0' in dfNew.columns:
                        dfNew.drop(columns=['Unnamed: 0'], inplace=True)
                    self.cleanData(dfNew)
                    df = pd.concat([dfNew, df], axis=0)
                    df.reset_index(drop=True, inplace=True)
                    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                    df.drop_duplicates(subset=['Date'], keep='first', inplace=True)
                    df.to_csv(f'{self.directory}/{component}.csv')
                    print(f'{component} data updated')
                    os.remove('new.csv')
                else:
                    continue

    def printData(self, components, start, end):
        startDate = pd.to_datetime(start)
        endDate = pd.to_datetime(end)
        for component in components:
            self.getData(start, end)
            df = pd.read_csv(f'{self.directory}/{component}.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            filtered_df = df[(df['Date'] >= startDate) & (df['Date'] <= endDate)]
            if 'Unnamed: 0' in filtered_df.columns:
                filtered_df.drop(columns=['Unnamed: 0'], inplace=True)
            filtered_df.reset_index(drop=True, inplace=True)
            print(f'-------------------{component}-------------------')
            print(filtered_df)

    def returnData(self, components, start, end):
        startDate = pd.to_datetime(start)
        endDate = pd.to_datetime(end)
        for component in components:
            self.getData(start, end)
            df = pd.read_csv(f'{self.directory}/{component}.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            filtered_df = df[(df['Date'] >= startDate) & (df['Date'] <= endDate)]
            if 'Unnamed: 0' in filtered_df.columns:
                filtered_df.drop(columns=['Unnamed: 0'], inplace=True)
            filtered_df.reset_index(drop=True, inplace=True)
            return filtered_df

class StatisticalAnalysis:
    def __init__(self, directory, start, end):
        self.directory = directory
        self.start = start
        self.end = end
        self.stockData = self.loadStockData()
        self.pairs = self.generatePairs()

    def loadStockData(self):
        stockData = {}
        for file in os.listdir(self.directory):
            if file.endswith('.csv'):
                stock = file.split('.')[0]
                df = pd.read_csv(f'{self.directory}/{file}', usecols=['Date', 'Adj Close'])
                df['Date'] = pd.to_datetime(df['Date'])
                df.sort_values(by='Date', inplace=True)
                df.reset_index(drop=True, inplace=True)
                stockData[stock] = df
        return stockData

    def generatePairs(self):
        return list(combinations(self.stockData.keys(), 2))

    def calculateCorrelation(self):
        results = []
        for pair in self.pairs:
            stock1, stock2 = pair
            df1 = self.stockData[stock1]
            df2 = self.stockData[stock2]
            merge = pd.merge(df1, df2, on='Date', suffixes=('_1', '_2'))
            correlation = merge['Adj Close_1'].corr(merge['Adj Close_2'])
            results.append({'Pair': pair, 'Correlation': correlation})
        return pd.DataFrame(results)

    def calculateCointegration(self):
        results = []
        for pair in self.pairs:
            stock1, stock2 = pair
            df1 = self.stockData[stock1]
            df2 = self.stockData[stock2]
            merge = pd.merge(df1, df2, on='Date', suffixes=('_1', '_2'))
            score, pvalue, _ = coint(merge['Adj Close_1'], merge['Adj Close_2'])
            results.append({'Pair': pair, 'P-Value': pvalue, 'Score': score})
        return pd.DataFrame(results)

    def optimalPairs(self):
        cointegration = self.calculateCointegration()
        correlation = self.calculateCorrelation()
        cointegration['Correlation'] = correlation['Correlation']
        return cointegration[(cointegration['P-Value'] < 0.05) & (cointegration['Correlation'] > 0.8)]

    def plot_heatmap(self):
        optimal = self.optimalPairs()
        data = optimal.pivot(index='Pair', columns='P-Value', values='Correlation')
        plt.figure(figsize=(10, 6))
        sns.heatmap(data, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Cointegration and Correlation Heatmap')
        plt.show()

    def plot_pair_prices(self, stock1, stock2):
        """Return a figure for pair prices instead of showing it."""
        df1 = self.stockData.get(stock1)
        df2 = self.stockData.get(stock2)
        if df1 is None or df2 is None:
            return None

        merge = pd.merge(df1, df2, on='Date', suffixes=('_1', '_2'))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(merge['Date'], merge['Adj Close_1'], label=stock1)
        ax.plot(merge['Date'], merge['Adj Close_2'], label=stock2)
        ax.legend()
        ax.set_title(f'Price Comparison: {stock1} vs {stock2}')
        return fig  # Return figure instead of calling plt.show()

    def plot_residuals(self, stock1, stock2):
        """Return a figure for residuals instead of showing it."""
        df1 = self.stockData.get(stock1)
        df2 = self.stockData.get(stock2)
        if df1 is None or df2 is None:
            return None

        merge = pd.merge(df1, df2, on='Date', suffixes=('_1', '_2'))
        residuals = merge['Adj Close_1'] - merge['Adj Close_2']
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(merge['Date'], residuals)
        ax.axhline(residuals.mean(), color='r', linestyle='--')
        ax.set_title(f'Residuals: {stock1} - {stock2}')
        return fig  # Return figure

    def plot_zscore(self, stock1, stock2):
        """Return a figure for Z-score instead of showing it."""
        df1 = self.stockData.get(stock1)
        df2 = self.stockData.get(stock2)
        if df1 is None or df2 is None:
            return None

        merge = pd.merge(df1, df2, on='Date', suffixes=('_1', '_2'))
        residuals = merge['Adj Close_1'] - merge['Adj Close_2']
        z_scores = (residuals - residuals.mean()) / residuals.std()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(merge['Date'], z_scores, label='Z-Score')
        ax.axhline(1, color='r', linestyle='--', label='Entry Threshold')
        ax.axhline(-1, color='g', linestyle='--', label='Exit Threshold')
        ax.legend()
        ax.set_title(f'Z-Score: {stock1} - {stock2}')
        return fig  # Return figure


class Position(Enum):
    NONE = 0
    LONG = 1
    SHORT = -1

class BackTest:
    def __init__(self, stock1, stock2, stockData, zentry=0.15, zexit=0, initial_capital=100000, risk_per_trade=0.2):
        self.stock1 = stock1
        self.stock2 = stock2
        self.stockData = stockData
        self.zentry = zentry
        self.zexit = zexit
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.residuals = None
        self.trades = []
        self.pnl_history = []
        self.trade_sizes = None

    def calculate_residuals(self):
        df1 = self.stockData[self.stock1]
        df2 = self.stockData[self.stock2]

        merged = pd.merge(df1, df2, on='Date', suffixes=('_1', '_2'))
        self.residuals = merged['Adj Close_1'].values - merged['Adj Close_2'].values
        self.prices1 = merged['Adj Close_1'].values
        self.prices2 = merged['Adj Close_2'].values
        self.dates = merged['Date'].values

    def calculate_ib_fees(self, price, shares):
        """Calculate Interactive Brokers' transaction fees."""
        fixed_fee = np.maximum(0.005 * shares, 1.00)  # $0.005 per share, min $1 per trade
        percentage_fee = np.maximum(0.000035 * (price * shares), 1.00)  # 0.0035% of trade value, min $1
        return np.maximum(fixed_fee, percentage_fee)

    def calculate_trade_sizes(self):
        """Dynamically calculate trade size based on capital & risk percentage."""
        risk_amount = self.current_capital * self.risk_per_trade  # Capital risk per trade
        max_trade_size = 0.30 * self.current_capital  # Max 10% of capital per trade

        trade_size1 = np.minimum((risk_amount // self.prices1).astype(int), max_trade_size // self.prices1)
        trade_size2 = np.minimum((risk_amount // self.prices2).astype(int), max_trade_size // self.prices2)

        self.trade_sizes = np.maximum(trade_size1, 1)  # Ensure at least 1 share is traded

    def run(self):
        """Run the backtesting logic."""
        if self.residuals is None:
            self.calculate_residuals()

        mean = np.mean(self.residuals)
        std = np.std(self.residuals)
        z_scores = (self.residuals - mean) / std

        self.calculate_trade_sizes()

        position = Position.NONE
        entry_price1, entry_price2 = 0, 0
        pnl = 0

        for i, z in enumerate(z_scores):
            if self.current_capital < self.initial_capital * 0.1:
                print("🚨 Trading stopped: Capital too low.")
                break

            trade_size1 = self.trade_sizes[i]
            trade_size2 = self.trade_sizes[i]

            if position == Position.NONE:
                if z > self.zentry:
                    position = Position.SHORT
                    entry_price1, entry_price2 = self.prices1[i], self.prices2[i]
                    fees = self.calculate_ib_fees(entry_price1, trade_size1) + self.calculate_ib_fees(entry_price2, trade_size2)
                    self.current_capital -= fees

                elif z < -self.zentry:
                    position = Position.LONG
                    entry_price1, entry_price2 = self.prices1[i], self.prices2[i]
                    fees = self.calculate_ib_fees(entry_price1, trade_size1) + self.calculate_ib_fees(entry_price2, trade_size2)
                    self.current_capital -= fees

            elif position == Position.SHORT and z <= self.zexit:
                pnl = (entry_price1 - self.prices1[i]) * trade_size1 + (self.prices2[i] - entry_price2) * trade_size2
                fees = self.calculate_ib_fees(self.prices1[i], trade_size1) + self.calculate_ib_fees(self.prices2[i], trade_size2)
                pnl -= fees

                # 🚨 Apply Stop-Loss
                max_loss_per_trade = 0.03 * self.current_capital
                if pnl < -max_loss_per_trade:
                    pnl = -max_loss_per_trade  # Cap the loss

                self.trades.append(pnl)
                self.pnl_history.append((self.dates[i], self.current_capital, pnl))
                self.current_capital += pnl
                position = Position.NONE

            elif position == Position.LONG and z >= -self.zexit:
                pnl = (self.prices1[i] - entry_price1) * trade_size1 + (entry_price2 - self.prices2[i]) * trade_size2
                fees = self.calculate_ib_fees(self.prices1[i], trade_size1) + self.calculate_ib_fees(self.prices2[i], trade_size2)
                pnl -= fees

                # 🚨 Apply Stop-Loss
                max_loss_per_trade = 0.03 * self.current_capital
                if pnl < -max_loss_per_trade:
                    pnl = -max_loss_per_trade  # Cap the loss

                self.trades.append(pnl)
                self.pnl_history.append((self.dates[i], self.current_capital, pnl))
                self.current_capital += pnl
                position = Position.NONE

        return self.current_capital, self.pnl_history

    def sharpe_ratio(self, risk_free_rate=0.02):
        """Calculate the Sharpe Ratio to evaluate risk-adjusted returns."""
        if not self.trades:
            return np.nan  # No trades = No ratio

        returns = np.array(self.trades)
        mean_return = np.mean(returns)
        std_dev = np.std(returns)

        if std_dev == 0:
            return np.nan  # Avoid division by zero

        return (mean_return - risk_free_rate) / std_dev

    def max_drawdown(self):
        """Calculate the Max Drawdown to measure worst capital loss."""
        if not self.pnl_history:
            return np.nan  # No trades = No drawdown

        capital_values = [cap for _, cap, _ in self.pnl_history]
        peak = capital_values[0]
        max_dd = 0

        for cap in capital_values:
            peak = max(peak, cap)
            drawdown = (peak - cap) / peak
            max_dd = max(max_dd, drawdown)

        return max_dd

    def plot_pnl(self):
        """Return a figure for PnL instead of showing it."""
        if not self.pnl_history:
            return None

        dates, capital_values, trade_pnl = zip(*self.pnl_history)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(dates, capital_values, label="Equity Curve", color="blue")
        ax.axhline(self.initial_capital, linestyle="--", color="gray", label="Starting Capital")
        ax.set_title("Equity Curve (Capital Over Time)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Capital ($)")
        ax.legend()
        return fig  # Return figure


stat = StatisticalAnalysis('stock data', '2024-01-01', '2025-01-01')
bt = BackTest('BA', 'WBA', stat.stockData, zentry=0.15, zexit=0, initial_capital=100000, risk_per_trade=0.5)
final_capital, trade_history = bt.run()
print(f"Final Capital: ${final_capital:.2f}")
for i, trade in enumerate(bt.pnl_history):
    print(f"Trade {i+1}: Date={trade[0]}, Capital={trade[1]:.2f}, PnL={trade[2]:.2f}")

print(f"Final Capital: ${final_capital:.2f}")
print(f"📊 Sharpe Ratio: {bt.sharpe_ratio():.2f}")
print(f"📉 Max Drawdown: {bt.max_drawdown()*100:.2f}%")



