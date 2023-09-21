import pandas as pd
from sklearn.metrics import mean_squared_error
import yfinance as yf




# Reset index and keep only 'Date' and 'Close' columns


# Backtest on ARIMA trained with close, currently only on AAPL
def backtest_model(model, start, end):
    data = yf.download('AAPL', start=start, end=end, progress=False)
    data = data.reset_index()[['Date', 'Close']]
    # Ensure the data is sorted by date in ascending order
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by='Date')

    # Create a new column for the predicted prices
    data['Predicted_Price'] = None

        # Prepare the testing data
    X_test = data[['Price']]

    # Use the pre-fitted model to make predictions
    predicted_prices = model.predict(X_test)

    # Store the predicted price
    data['Predicted_Price'] = predicted_prices

    # Calculate performance metrics
    mse = mean_squared_error(data['Price'].shift(-1).dropna(), data['Predicted_Price'].dropna())
    return mse

