#Using RWH, we can create x days of lagged stock prices. 
#We then use OLS Regression to predict todays prices. 
import numpy as np
import os
import pandas as pd
from pylab import mpl, plt
from polygon import RESTClient
from itertools import product
from sklearn.linear_model import LinearRegression
os.environ['apikey'] = ''
client = RESTClient(api_key=os.environ['apikey'])
ticker = 'MSFT'

stockData = list()
for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="minute", from_="2023-06-01", to="2023-06-13", limit=50000):
    stockData.append(a.close)

data = pd.DataFrame(stockData)
lags = 5
lagsColums = list()
for lag in range(1, lags + 1):
    col = "lag: {}".format(lag)
    data[col] = data[0].shift(lag)
    lagsColums.append(col)

data.dropna(inplace=True)
data['returns'] = np.log(data[0] / data[0].shift(1))
data.dropna(inplace=True)
data['direction'] = np.sign(data['returns']).astype(int)
data.dropna(inplace=True)

#Linear OLS Regression 
model = LinearRegression()
data['firstOLS'] = model.fit(data[lagsColums],data['returns']).predict(data[lagsColums])
data['secondOLS'] = model.fit(data[lagsColums],data['direction']).predict(data[lagsColums])
data[['firstOLS', 'secondOLS']] = np.where(data[['firstOLS', 'secondOLS']] > 0, 1, -1)

#vectorized backtesting
data['predictionFirstOLS'] = data['firstOLS'] * data['returns']
data['predictionSecondOLS'] = data['secondOLS'] * data['returns']
data[['returns', 'predictionFirstOLS', 'predictionSecondOLS']].sum().apply(np.exp)
data[['returns', 'predictionFirstOLS', 'predictionSecondOLS']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()

