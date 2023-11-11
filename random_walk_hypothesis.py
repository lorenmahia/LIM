#Using RWH, we can create x days of lagged stock prices. 
#We then use OLS Regression to predict todays prices. 
import numpy as np
import os
import pandas as pd
from pylab import mpl, plt
from polygon import RESTClient
from itertools import product
from sklearn.linear_model import LinearRegression
os.environ['apikey'] = 'vf1yK4D8c5Opbw3RKDwOqli0Rjmtzdx0'
client = RESTClient(api_key=os.environ['apikey'])
ticker = 'MSFT'

stockData = list()
for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="minute", from_="2023-06-01", to="2023-06-13", limit=50000):
    stockData.append(a.close)

data = pd.DataFrame(stockData)
lags = 5
cols = list()
for lag in range(1, lags + 1):
    col = "lag: {}".format(lag)
    data[col] = data[0].shift(lag)
    cols.append(col)

data.dropna(inplace=True)
data['returns'] = np.log(data[0] / data[0].shift(1))
data.dropna(inplace=True)
data['direction'] = np.sign(data['returns']).astype(int)
data.dropna(inplace=True)

#Linear OLS Regression 
model = LinearRegression()
data['pos_ols_1'] = model.fit(data[cols],data['returns']).predict(data[cols])
data['pos_ols_2'] = model.fit(data[cols],data['direction']).predict(data[cols])
data[['pos_ols_1', 'pos_ols_2']] = np.where(data[['pos_ols_1', 'pos_ols_2']] > 0, 1, -1)

#vectorized backtesting
data['strat_ols_1'] = data['pos_ols_1'] * data['returns']
data['strat_ols_2'] = data['pos_ols_2'] * data['returns']
data[['returns', 'strat_ols_1', 'strat_ols_2']].sum().apply(np.exp)
data[['returns', 'strat_ols_1', 'strat_ols_2']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()

