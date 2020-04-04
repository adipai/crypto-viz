from flask import Flask
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model

def model_definition():
    model= Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(1,1)))
    model.add(LSTM(256))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

eth_model = model_definition()
eth_model.load_weights("cryptoviz/models/eth_model.h5")
eth_model._make_predict_function()

btc_model = model_definition()
btc_model.load_weights("cryptoviz/models/btc_model.h5")
btc_model._make_predict_function()

ltc_model = model_definition()
ltc_model.load_weights("cryptoviz/models/ltc_model.h5")
ltc_model._make_predict_function()

xrp_model = model_definition()
xrp_model.load_weights("cryptoviz/models/xrp_model.h5")
xrp_model._make_predict_function()

flaskapp = Flask(__name__)

import cryptoviz.views
