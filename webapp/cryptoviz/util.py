import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from binance.client import Client

def data_extract(sym, start, end):
    client = Client('API_KEY', 'SECRET_KEY')
    if end =="":
        CRYPTOCURRENCY = client.get_historical_klines(symbol=sym, interval=Client.KLINE_INTERVAL_30MINUTE, start_str=start)
    else:
        CRYPTOCURRENCY = client.get_historical_klines(symbol=sym, interval=Client.KLINE_INTERVAL_30MINUTE, start_str=start, end_str=end)

    CRYPTOCURRENCY = pd.DataFrame(CRYPTOCURRENCY, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    CRYPTOCURRENCY['Open time'] = pd.to_datetime(CRYPTOCURRENCY['Open time'], unit='ms')
    CRYPTOCURRENCY.set_index('Open time', inplace=True)
    #print(CRYPTOCURRENCY.head())
    return CRYPTOCURRENCY.iloc[:,3:4].astype(float).values

def plot_graph(crypto, predicted_price, real_price):
    img = io.BytesIO()
    plt.figure(figsize=(10,4))
    red_patch = mpatches.Patch(color='red', label='Predicted Price of {}'.format(crypto))
    blue_patch = mpatches.Patch(color='blue', label='Real Price of {}'.format(crypto))
    plt.legend(handles=[blue_patch, red_patch])
    plt.plot(predicted_price, color='red', label='Predicted Price of {}'.format(crypto))
    plt.plot(real_price, color='blue', label='Predicted Price of {}'.format(crypto))
    plt.title('Predicted vs. Real Price of {}'.format(crypto))
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    #plt.savefig('{}.png'.format(crypto))
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()
