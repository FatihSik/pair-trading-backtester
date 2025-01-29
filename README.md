# Pair Trading Backtester

## Overview
This project is a **pair trading backtesting system** with a graphical user interface (GUI) built using **Tkinter**. The system allows users to select stock pairs from major indices (e.g., S&P 500, DJIA, and Crypto), perform statistical analysis, and run backtests to evaluate trading strategies.

## Features
- **GUI for selecting indices and trading pairs**
- **Automatic fetching of stock data using Yahoo Finance (`yfinance`)**
- **Statistical analysis including correlation and cointegration tests**
- **Graphical visualization of equity curves, price movements, residuals, and Z-scores**
- **Backtesting with capital management, risk per trade settings, and transaction cost simulation**
- **Performance evaluation using Sharpe Ratio and Max Drawdown**

## Installation
### Prerequisites
Ensure you have **Python 3.x** installed along with the following libraries:

```sh
pip install yfinance pandas numpy statsmodels matplotlib seaborn
```

### Clone the Repository
```sh
git clone https://github.com/FatihSik/pair-trading-backtester.git
cd pair-trading-backtester
```

## Usage
1. **Run the application:**
    ```sh
    python app.py
    ```
2. **Select an index (e.g., S&P 500, DJIA, Crypto)**
3. **Choose a pair from the suggested optimal pairs**
4. **Click "Run Backtest" to evaluate the pair trading strategy**
5. **View results including final capital, trade history, and performance metrics**
6. **Visualize key metrics like equity curves and residual analysis**

## File Structure
```
├── app.py                # GUI application for running backtests
├── main.py               # Statistical analysis & backtesting logic
├── sandp.csv             # S&P 500 index components
├── djia.csv              # Dow Jones index components
├── crypto.csv            # Cryptocurrency index components
├── stock data/           # Directory for storing fetched stock price data
```

## Backtesting Strategy
The backtesting model uses:
- **Cointegration tests** to identify mean-reverting pairs
- **Z-score-based trading rules** for entry and exit
- **Risk management**, including position sizing and stop-loss
- **Transaction cost modeling** based on Interactive Brokers’ fee structure

## Example Output
```
Final Capital: $105,200.45
📊 Sharpe Ratio: 1.45
📉 Max Drawdown: 6.32%
```

## Contributing
Pull requests are welcome! If you have ideas for improvements, please create an issue or fork the repository.

## License
This project is open-source and available under the MIT License.

