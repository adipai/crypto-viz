# crypto-viz
Cryptocurrency price prediction using LSTM with flask webapp deployment.

## Requirements
* binance api
```
$ python -m pip install python-binance
```
* tensorflow 1.x / keras 2.2.5 
* flask
* numpy
* pandas
* matplotlib

## Running the webapp
```
$ cd webapp
$ python3 app.py
Open your internet browser and search for "localhost:5000" in the URL box.
```
## Snapshots
### Homepage
![homepage](https://github.com/adipai/crypto-viz/blob/master/images/results/webapp.JPG)
### Forecasts for BTC (01/01/2020-01/02/2020)
![btc forecast](https://github.com/adipai/crypto-viz/blob/master/images/results/bitcoin.JPG)
### Forecasts for ETH (01/01/2020-01/02/2020)
![eth forecast](https://github.com/adipai/crypto-viz/blob/master/images/results/ethereum.JPG)
### Forecasts for LTC (01/01/2020-01/02/2020)
![eth forecast](https://github.com/adipai/crypto-viz/blob/master/images/results/litecoin.JPG)
### Forecasts for XRP (01/01/2020-01/02/2020)
![eth forecast](https://github.com/adipai/crypto-viz/blob/master/images/results/xrp.JPG)

### Realtime deployment
* add/replace trained models in webapp/cryptoviz/models/
* use API intergration for extracting prices until current timestamp to predict for future(binance lets you extract prices until 3 hours ago...best case)
* automate the prediction for continous forecast
