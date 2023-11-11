import numpy as np
import os
import pandas as pd
import datetime as dt
from pylab import mpl, plt
from polygon import RESTClient
from itertools import product
os.environ['apikey'] = 'vf1yK4D8c5Opbw3RKDwOqli0Rjmtzdx0'
client = RESTClient(api_key=os.environ['apikey'])
ticker = 'MSFT'

stockData = list()
for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="minute", from_="2023-06-01", to="2023-06-13", limit=50000):
    stockData.append(a.close)


data = (pd.DataFrame(stockData).dropna())

#Calculate Simple moving averages 
SMA1 = 42
SMA2 = 252

data['SMA1'] = data[0].rolling(SMA1).mean()
data['SMA2'] = data[0].rolling(SMA2).mean()
data.plot(figsize=(10, 6))
#plt.show()

#The Trading Rules 
#Go long (= +1) when the shorter SMA is above the longer SMA.
#Go short (= -1) when the shorter SMA is below the longer SMA.
data.dropna(inplace=True)
#Places 1 when we are long and -1 otherwise
data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
data.tail()

ax = data.plot(secondary_y='Position', figsize=(10, 6))
ax.get_legend().set_bbox_to_anchor((0.25, 0.85))

#Vectorized BackTesting 
#Calculating Log Stock Returns
data['Returns'] = np.log(data[0] / data[0].shift(1))
#Shifts position values by 1 day and multiply by log returns
data['Strategy'] = data['Position'].shift(1) * data['Returns']
data.dropna(inplace=True)
#Exponential value to arrive at peak performance 
np.exp(data[['Returns', 'Strategy']].sum())
#Calculate Annualized volatility 
data[['Returns', 'Strategy']].std() * 252 ** 0.5

#Model performance over stock performance
ax = data[['Returns', 'Strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
data['Position'].plot(ax=ax, secondary_y='Position', style='--')
ax.get_legend().set_bbox_to_anchor((0.25, 0.85)); 


#Brute Force Method to Optimize Returns
sma1 = range(20,64,3)
sma2 = range(180,281,11)

results = pd.DataFrame()
for SMA,SMA2 in product(sma1,sma2):
    data = pd.DataFrame(stockData)
    data['Returns'] = np.log(data[0] / data[0].shift(1))
    data['SMA1'] = data[0].rolling(SMA1).mean()
    data['SMA2'] = data[0].rolling(SMA2).mean()
    data.dropna(inplace=True)
    data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
    data['Strategy'] = data['Position'].shift(1) * data['Returns']
    data.dropna(inplace=True)
    perf = np.exp(data[['Returns', 'Strategy']].sum())
    results = results._append(pd.DataFrame(
    {'SMA1': SMA1, 'SMA2': SMA2,
    'MARKET': perf['Returns'],
    'STRATEGY': perf['Strategy'],
    'OUT': perf['Strategy'] - perf['Returns']},
    index=[0]), ignore_index=True)

#sort first 7 values
results.sort_values('OUT', ascending=False).head(7)
print(results)