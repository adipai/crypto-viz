import datetime
from binance.client import Client
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys


"""
This script is for prediction only. We already have the pre-trained model with weights in model.json and model h5 respectively,
we have to just load it and predict. We can make this script the backend for the static website/app/CLI.
"""
# Instanciate Binance client
client = Client('API_KEY', 'SECRET_KEY')

"""
# Just a tried for CLI. Simple but boring :-P
# Run the script using > python predict.py <CRYPTOCURRENCY>, eg. python predict.py bitcoin
if sys.argv[1] == 'bitcoin':
    symbol = 'BTCUSDT'
elif sys.argv[1] == 'ethereum':
    symbol = 'ETHUSDT'
elif sys.argv[1] == 'ripple':
    symbol = 'XRPUSDT'
elif sys.argv[1] == 'litecoin':
    symbol = 'LTCUSDT'
else:
    print(sys.argv[1]+' doesn\'t exist or isn\'t implemented yet.')
    sys.exit()
"""

# Same pipeline as the training file here !
# get data
symbol = 'BTCUSDT'
CRYPTOCURRENCY = client.get_historical_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_30MINUTE, start_str="1 year ago UTC")
CRYPTOCURRENCY = pd.DataFrame(CRYPTOCURRENCY, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])

CRYPTOCURRENCY['Open time'] = pd.to_datetime(CRYPTOCURRENCY['Open time'], unit='ms')

CRYPTOCURRENCY.set_index('Open time', inplace=True)

CRYPTOCURRENCY['Close']=CRYPTOCURRENCY['Close'].astype(float)

data = CRYPTOCURRENCY.iloc[:,3:4].astype(float).values

# Scale data
scaler= MinMaxScaler()
data= scaler.fit_transform(data)

# We dont need the training_set here, only passing test... but just for the size paremeter :-P
training_set = data[:10000]
test_set = data[10000:]
# Data preprocessing (Dividing datasets to training and testing data)
X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]

X_test = test_set[0:len(test_set)-1]
y_test = test_set[1:len(test_set)]

X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))

#model = load_model(sys.argv[1]+'_model.h5')
#model = load_model('bitcoin_model.h5')

# load the model from json file, gives an error for my version of keras, works on on colab tho..seems to be happening for LSTM, worked fine for CNN :-(

"""json_file = open('model.json', 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)"""

# So i just defined the architecture again here, not needed if the above snippet works
model= Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(256))
model.add(Dense(1))

# the pre-trained weights are loaded from model.h5 into the our model
model.load_weights("model.h5")
print("Loaded model from disk")
model.compile(loss='mean_squared_error', optimizer='adam')

# We're good to go now, no training required cuz we loaded the pre-trained model

# Perform predictions on test data
predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)
real_price = scaler.inverse_transform(y_test)

# Display graph of our prediction
"""
plt.figure(figsize=(10,4))
red_patch = mpatches.Patch(color='red', label='Predicted Price of '+sys.argv[1])
blue_patch = mpatches.Patch(color='blue', label='Real Price of '+sys.argv[1])
plt.legend(handles=[blue_patch, red_patch])
plt.plot(predicted_price, color='red', label='Predicted Price of '+sys.argv[1])
plt.plot(real_price, color='blue', label='Real Price of '+sys.argv[1])
plt.title('Predicted vs. Real Price of '+sys.argv[1])
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
"""
plt.figure(figsize=(10,4))
red_patch = mpatches.Patch(color='red', label='Predicted Price of bitcoin')
blue_patch = mpatches.Patch(color='blue', label='Real Price of bitcoin')
plt.legend(handles=[blue_patch, red_patch])
plt.plot(predicted_price, color='red', label='Predicted Price of bitcoin')
plt.plot(real_price, color='blue', label='Real Price of bitcoin')
plt.title('Predicted vs. Real Price of bitcoin')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
