''''
This Code adds Machine Learning to the Kelly Fraction Strategy to maximize returns. 
All data is binarized and a supervised training model is used. 
'''
import numpy as np
import os
import math
import pickle
import time
import pandas as pd
from pylab import mpl, plt
from polygon import RESTClient
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

os.environ['apikey'] = ''
client = RESTClient(api_key=os.environ['apikey'])
ticker = 'SPY'

stockData = list()
for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="day", from_="2021-06-01", to="2023-06-13", limit=50000):
    stockData.append(a.close)

data = pd.DataFrame(stockData)
data["returns"] = np.log(data[0] / data[0].shift(1))
data.dropna(inplace=True)
lags = 5
lagsColumns = list()
for lag in range(1, lags + 1):
    col = "lag: {}".format(lag)
    data[col] = data[0].shift(lag)
    lagsColumns.append(col)

#Binarizing data 
data[lagsColumns] = np.where(data[lagsColumns] > 0 , 1, 0)
data["direction"] = np.where(data["returns"] > 0, 1, -1)

''''
Using Machine learning based on the support vector machine(SVM) that had superior 
performance in model development
Since performing a trade generates tranasction costs, this model accounts for this, producing 
better results than the random walk hypothesis
'''
model = SVC(C = 1, kernel="linear", gamma="auto")
#Spliting the data for Train trainTest Modelling
dataSplit = (int)(len(data) * 0.8)
trainTest = data.iloc[:dataSplit].copy()
model.fit(trainTest[lagsColumns],trainTest["direction"])
#Accuracy score of the training data 
trainingScore = accuracy_score(trainTest["direction"],model.predict(trainTest[lagsColumns]))
#Accuracy score of the trainTest data
trainTest["position"] = model.predict(trainTest[lagsColumns])
testScore = accuracy_score(trainTest["direction"],trainTest["position"])

#Derives log returns for this strategy
trainTest["strategy"] = trainTest["position"] * trainTest["returns"]
#Including transaction costs 
tranasctionCost = 0.06 #TODO
trainTest["strategywCosts"] = np.where(trainTest["position"].diff() != 0, trainTest["strategy"] - tranasctionCost, trainTest["strategy"])
trainTest[['returns', 'strategy', 'strategywCosts']].cumsum(
).apply(np.exp).plot(figsize=(10, 6))

#Maximizing Profits by optimizing values 
#Annualized Mean ,Variance and Volatility Returns
testDataMean = trainTest[["returns", "strategywCosts"]].mean() * len(data) * 12
testDataVariance = trainTest[["returns", "strategywCosts"]].var() * len(data) *12
testDataVolatility = testDataVariance ** 0.5

#Fully Kelly Criterion
fullKelly = testDataMean / testDataVariance
#Half Kelly Criterion 
halfKelly = testDataMean / testDataVariance * 0.5

#Scaling Leverages 
#CAUTION: Scaling Leverages increases Risk and should be used wisely
scaledLeverages = ['returns', 'strategywCosts']
for testLeverage in [5, 10, 15]:
    label = 'strategywCosts_%d' % testLeverage
    trainTest[label] = trainTest['strategywCosts'] * testLeverage
    scaledLeverages.append(label)
trainTest[scaledLeverages].cumsum().apply(np.exp).plot(figsize=(10, 6))

#Risk analysis to reduce leverage associated risks
#TODO

#Serializing the Model for Deployment 
pickle.dump(model,open('LIGMLModel.pk','wb'))
    
