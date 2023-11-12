#Using Classification Methods in ML to predict stock values
#We will use three models, logistic regression, Gaussian Naive Bayes, and support vector machine
import numpy as np
import os
import pandas as pd
from pylab import mpl, plt
from polygon import RESTClient
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

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
data['returns'] = np.log(data[0] / data[0].shift(1))
data.dropna(inplace=True)
data['direction'] = np.sign(data['returns']).astype(int)
data.dropna(inplace=True)

#Two Binary Features 
C = 1
models = {
    'log_reg': linear_model.LogisticRegression(C=C),
    'gauss_nb': GaussianNB(),
    'svm': SVC(C=C)
}

#Function that fits all models 
def fit_models(data):
    mfit = {model: models[model].fit(data[cols_bin],data['direction'])
            for model in models.keys()}

#Function to derive all position values from fitted models
def derive_positions(data):
    for model in models.keys():
        data['pos_' + model] = models[model].predict(data[cols_bin])
#Function to evaluate all trading strategies
def evaluate_strategies(data):
    global sel
    sel = []
    for model in models.keys():
        col = 'strat_' + model
        data[col] = data['pos_' + model] * data['returns']
        sel.append(col)
    sel.insert(0, 'returns')
#Function to create testing bins 
def create_bins(data, bins=[0]):
    global cols_bin
    cols_bin = []
    for col in lagsColumns:
        col_bin = col + '_bin'
        data[col_bin] = np.digitize(data[col], bins=bins)
        cols_bin.append(col_bin)

create_bins(data)
fit_models(data)
derive_positions(data)
evaluate_strategies(data)
data[sel].sum().apply(np.exp)
data[sel].cumsum().apply(np.exp).plot(figsize=(10, 6))

#Five Digitized Features
meanReturns = data['returns'].mean()
stdReturns = data['returns'].std()
#To digitize the featured data 
bins = [meanReturns - stdReturns, meanReturns, meanReturns + stdReturns]

create_bins(data,bins)
fit_models(data)
derive_positions(data)
evaluate_strategies(data)
data[sel].sum().apply(np.exp)
data[sel].cumsum().apply(np.exp).plot(figsize=(10, 6))

'''''
Sequential Train Test Split 
Sequential test train (STT) is a technique for evaluating the performance of machine learning models. 
It is an iterative process that involves training a model on a subset of the data, evaluating it on a held-out test set, 
and then adding more data to the training set and repeating the process.
This process is continued until the desired level of performance is reached.
'''''
split = int(len(data) * 0.5)
train = data.iloc[:split].copy()
fit_models(train)
test = data.iloc[split:].copy()
derive_positions(test)
evaluate_strategies(test)
test[sel].sum().apply(np.exp)
test[sel].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()


