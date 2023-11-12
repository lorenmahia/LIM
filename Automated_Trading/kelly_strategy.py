''''
The Kelly criterion is based on the idea that investors should allocate more money to bets with a higher
probability of success and a higher expected return.
The formula assums a binomial model where stock prices can only go in two directions

The formula for the Kelly criterion is:

f = (b - 1) / R
where:
f is the fraction of the investor's bankroll that should be invested in the bet
b is the expected return of the bet
R is the ratio of the expected return to the variance of the return
'''
import numpy as np
import os
import math
import pandas as pd
from pylab import mpl, plt
from polygon import RESTClient

os.environ['apikey'] = ''
client = RESTClient(api_key=os.environ['apikey'])
ticker = 'SPY'

stockData = list()
for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="day", from_="2021-06-01", to="2023-06-13", limit=50000):
    stockData.append(a.close)

data = pd.DataFrame(stockData)
data["returns"] = np.log(data[0] / data[0].shift(1))
data.dropna(inplace=True)
meanReturns = data["returns"].mean() * 252
volatility = data["returns"].std()*252 ** 0.5
riskFreeRate = 0
kellyFraction = (meanReturns - riskFreeRate) / volatility **2
equs = list()

''''
A function that implements the Kelly Strategy 
'''
def kelly_strategy(kellyFraction):
    global equs
    equ = 'equity_{:.2f}'.format(kellyFraction)
    equs.append(equ)
    cap = 'capital_{:.2f}'.format(kellyFraction)
    data[equ] = 1 #For simplicity original equity is set to 1
    data[cap] = data[equ] * kellyFraction
    for i, t in enumerate(data[0].index[1:]):
        t_1 = data.index[i]
        #New capital position given return 
        data.loc[t, cap] = data[cap].loc[t_1] * math.exp(data['returns'].loc[t])
        #Adjust equity value with performance
        data.loc[t, equ] = data[cap].loc[t] - data[cap].loc[t_1] + data[equ].loc[t_1]
        data.loc[t, cap] = data[equ].loc[t] * kellyFraction

#Ploting Equity values with different kelly fractions
kelly_strategy(kellyFraction * 0.5)
kelly_strategy(kellyFraction * 0.7)
kelly_strategy(kellyFraction * 0.065)
kelly_strategy(kellyFraction)

kellyPlots = data['returns'].cumsum().apply(np.exp).plot(legend=True,
figsize=(10, 6))
data[equs].plot(legend=True)
plt.show()