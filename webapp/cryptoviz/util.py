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
        cryptocurrency = client.get_historical_klines(symbol=sym, interval=Client.KLINE_INTERVAL_30MINUTE, start_str=start)
    else:
        cryptocurrency = client.get_historical_klines(symbol=sym, interval=Client.KLINE_INTERVAL_30MINUTE, start_str=start, end_str=end)

    cryptocurrency = pd.DataFrame(cryptocurrency, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    cryptocurrency['Open time'] = pd.to_datetime(cryptocurrency['Open time'], unit='ms')
    cryptocurrency.set_index('Open time', inplace=True)
    #print(CRYPTOCURRENCY.head())
    return cryptocurrency.iloc[:,3:4].astype(float).values

def plot_graph(crypto, predicted_price, real_price):
    img = io.BytesIO()
    plt.figure(figsize=(10,4))
    red_patch = mpatches.Patch(color='orange', label='Predicted Price of {}'.format(crypto))
    blue_patch = mpatches.Patch(color='purple', label='Real Price of {}'.format(crypto))
    plt.legend(handles=[blue_patch, red_patch])
    plt.plot(predicted_price, color='orange', label='Predicted Price of {}'.format(crypto))
    plt.plot(real_price, color='purple', label='Predicted Price of {}'.format(crypto))
    plt.title('Predicted vs. Real Price of {}'.format(crypto))
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    #plt.savefig('{}.png'.format(crypto))
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()
