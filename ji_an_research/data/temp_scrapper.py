import yfinance as yf
import pandas as pd

ticker_sector_mapping = {
    'XOM': 'Energy', 
    'CVX': 'Energy', 
    'SHW': 'Materials', 
    'DD': 'Materials', 
    'UPS': 'Industrials', 
    'RTX': 'Industrials',
    'DUK': 'Utilities', 
    'ED': 'Utilities', 
    'AEP': 'Utilities', 
    'UNH': 'Healthcare',
    'JNJ': 'Healthcare', 
    'BRK-A': 'Financials', 
    'BRK-B': 'Financials',
    'JPM': 'Financials', 
    'AMZN': 'Consumer Discretionary', 
    'MCD': 'Consumer Discretionary',
    'KO': 'Consumer Staples', 
    'PG': 'Consumer Staples', 
    'AAPL': 'Information Technology', 
    'MSFT': 'Information Technology', 
    'META': 'Communication Services', 
    'GOOGL': 'Communication Services',
    'AMT': 'Real Estate', 
    'SPG': 'Real Estate'
}

all_stocks = {}

for ticker, sector in ticker_sector_mapping.items():
    stock_data = yf.download(ticker)
    stock_data = stock_data.dropna()
    stock_data['Sector'] = sector  # Add the sector column
    stock_data['Ticker'] = ticker  # Add the ticker column
    all_stocks[ticker] = stock_data

# Save each ticker's data to a separate sheet in a single Excel file
with pd.ExcelWriter('data_for_testing.xlsx') as writer:
    for ticker, data in all_stocks.items():
        data.to_excel(writer, sheet_name=ticker)
