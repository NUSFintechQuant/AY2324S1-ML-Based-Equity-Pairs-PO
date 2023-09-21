import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# Functions for LSTM
class LstmBuilder():
    # 10 is not a good number of neurons, but it's a start for testing
    # 50 is a good number of neurons, but it takes a long time to train
    def __init__(self, neutrons=10, time_step = 60, loss="mse", batch_size = 1):
        self.neutrons = neutrons
        self.time_step = time_step
        self.loss = loss
        self.batch_size = batch_size


    def create_sequences(self, data):
        """
        Create sequences from the data.

        Parameters:
        - data: Original time series data.
        - time_steps: Number of time steps in each sequence.

        Returns:
        - X: Sequences
        - y: Targets (the subsequent values)
        """
        
        X, y = [], []

        for i in range(len(data) - self.time_step):
            # Extract the sequence and the subsequent value
            seq = data[i:(i + self.time_step)]
            target = data[i + self.time_step]
            
            X.append(seq)
            y.append(target)

        return np.array(X), np.array(y)

    def create_model(self, features):
        model = Sequential()
        model.add(LSTM(self.neutrons, activation='relu', input_shape=(self.time_step, features)))
        model.add(Dense(features))
        model.compile(optimizer='adam', loss=self.loss)
        return model

    def create_stateful_model(self, features):
        model = Sequential()
        model.add(LSTM(self.neutrons, activation='relu', batch_input_shape=(self.batch_size, self.time_step, features), stateful=True))
        model.add(Dense(features))
        model.compile(optimizer='adam', loss=self.loss)
        return model

    
    def split_data(self, X, y, size=0.7):
        train_size = int(size * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        return [X_train, X_test, y_train, y_test]

    
    def split_stateful_data(self, X, y, size=0.7):
        batch_size = self.batch_size
        train_size = int(size * len(X))

        # Adjust the train_size to be a multiple of batch_size
        train_size = (train_size // batch_size) * batch_size

        # Split data
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Adjust test data to be a multiple of batch_size
        test_size = (len(X_test) // batch_size) * batch_size
        X_test = X_test[:test_size]
        y_test = y_test[:test_size]

        return [X_train, X_test, y_train, y_test]
