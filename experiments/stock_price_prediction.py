import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

data = yf.download('AAPL', start='2010-01-01', end='2021-12-31')

close_prices = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=10)

test_data = scaled_data[train_size - time_step:]
X_test, y_test = create_dataset(test_data, time_step)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE:", rmse)

train = data['Close'][:train_size]
valid = data['Close'][train_size:]
valid = valid.to_frame()
valid['Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.plot(train)
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'])
plt.show()

def make_future_predictions(model, data, time_step, n_predictions):
    predictions = []
    last_sequence = data[-time_step:, :]
    for _ in range(n_predictions):
        last_sequence_reshaped = last_sequence.reshape(1, time_step, 1)
        next_prediction = model.predict(last_sequence_reshaped)
        predictions.append(next_prediction[0, 0])
        last_sequence = np.append(last_sequence[1:, :], next_prediction)
        last_sequence = last_sequence.reshape(time_step, 1)
    return np.array(predictions)

future_predictions = make_future_predictions(model, scaled_data, time_step, 30)
future_predictions_inverse = scaler.inverse_transform(future_predictions.reshape(-1, 1))

plt.plot(future_predictions_inverse)
plt.show()