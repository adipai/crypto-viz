
import datetime
from binance.client import Client
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

"""
Binance
-https://www.binance.com/en
-create an account, API management, create API, authenticate using google authenticater app/SMS
-you'll get an API key and secret key which will be shown only once, so copy it into a notepad and keep it safely
-you can just run the code directly, the above instructions are just for docs :-P
"""
# instanciate Binance client
client = Client('API_KEY', 'SECRET_KEY')
# symbol needs be replaced with '<cyptoname>USDT', e.g - ETHUSDT(ethereum),XRPUSDT(Ripple) etc..
symbol = 'BTCUSDT'
# data acquisition - candlestick data
BTC = client.get_historical_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_30MINUTE, start_str="1 year ago UTC")
"""
# fetch 1 minute klines for the last day up until now
klines = client.get_historical_klines("BNBBTC", Client.KLINE_INTERVAL_1MINUTE, start_str="1 day ago UTC")

# fetch 30 minute klines for the last month of 2017
klines = client.get_historical_klines("ETHBTC", Client.KLINE_INTERVAL_30MINUTE, start_str="1 Dec, 2017", end_str="1 Jan, 2018")

# fetch weekly klines since it listed
klines = client.get_historical_klines("NEOBTC", Client.KLINE_INTERVAL_1WEEK, start_str="1 Jan, 2017")
More info:
https://python-binance.readthedocs.io/en/latest/market_data.html
"""
# json to DataFrame
BTC = pd.DataFrame(BTC, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])

#convert time/date from unix timestamp to normal format (hh:mm:ss, mm/dd/yyyy)
BTC['Open time'] = pd.to_datetime(BTC['Open time'], unit='ms')

#make time the index of the DataFrame
BTC.set_index('Open time', inplace=True)

# Close will go into the rnn and be predicted
BTC['Close']=BTC['Close'].astype(float)
data = BTC.iloc[:,3:4].astype(float).values

# Scaling with minmax
scaler= MinMaxScaler()
data= scaler.fit_transform(data)

#first 10k elements (0-9999) are training
training_set = data[:10000]
#rest of them are test
test_set = data[10000:]
# Data preprocessing (Dividing datasets to training and testing data)
# we're doing this manually to avoid shuffling
X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]

X_test = test_set[0:len(test_set)-1]
y_test = test_set[1:len(test_set)]

# the confusing part tbh , the sequential data to tensor(matrix of n dim) ... here n=3
X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))

# Architecture of neural network
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(256))
model.add(Dense(1))
# Compile the model
# optimizer reduces loss fuction, i.e,  adam reduces mean_squared_error
model.compile(loss='mean_squared_error', optimizer='adam')
# train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, shuffle=False)

"""
Save the trained model so that it can be directly used later without training again, this is called transfer learning.
The model Architecture is stored in model.json, weights in model.h5
"""
# serialize model to JSO
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# Perform predictions on test data
predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)
real_price = scaler.inverse_transform(y_test)

# Display graph of our prediction
plt.figure(figsize=(10,4))
red_patch = mpatches.Patch(color='red', label='Predicted Price of Bitcoin')
blue_patch = mpatches.Patch(color='blue', label='Real Price of Bitcoin')
plt.legend(handles=[blue_patch, red_patch])
plt.plot(predicted_price, color='red', label='Predicted Price of Bitcoin')
plt.plot(real_price, color='blue', label='Real Price of Bitcoin')
plt.title('Predicted vs. Real Price of Bitcoin')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
