#The algo identifies feature clusters that determine upward or downward movement
import numpy as np
import os
import pandas as pd
from pylab import mpl, plt
from polygon import RESTClient
from itertools import product
from sklearn.cluster import KMeans

os.environ['apikey'] = ''
client = RESTClient(api_key=os.environ['apikey'])
ticker = 'MSFT'

stockData = list()
for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="minute", from_="2023-06-01", to="2023-06-13", limit=50000):
    stockData.append(a.close)

data = pd.DataFrame(stockData)

lags = 5
lagsColumns = list()
for lag in range(1, lags + 1):
    col = "lag: {}".format(lag)
    data[col] = data[0].shift(lag)
    lagsColumns.append(col)

data.dropna(inplace=True)
#Kmeans Algorithm 
model = KMeans(n_clusters=2,random_state=0)
model.fit(data[lagsColumns])
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
n_clusters=2, n_init=10,
random_state=0, tol=0.0001, verbose=0)
data['positionCluster'] = model.predict(data[lagsColumns])
data['positionCluster'] = np.where(data['positionCluster'] == 1, -1, 1)
plt.figure(figsize=(10, 6))
plt.scatter(data[lagsColumns].iloc[:, 0], data[lagsColumns].iloc[:, 1],c=data['positionCluster'], cmap='coolwarm')

#Vectorized Backtesting to observe reults 
data['returns'] = np.log(data[0] / data[0].shift(1))
data['predictionCluster'] = data['positionCluster'] * data['returns']
data[['returns', 'predictionCluster']].sum().apply(np.exp)
data[['returns', 'predictionCluster']].cumsum().apply(np.exp).plot(figsize=(10, 6))