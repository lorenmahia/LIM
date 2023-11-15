''''
Using RobinHood API to make all necessary calls to enable online trading
'''
import robin_stocks.robinhood as rh
import pandas as pd 
import time
import os
from pylab import plt
import numpy as np

os.environ['username'] = ''
os.environ['password'] = ''
login = rh.login(username=os.environ['username'],password=os.environ['password'])

#Accessing holdings
rh.build_holdings(True)

#Function to Visualize specific stock ticker prices
def visualize_price (stockTickers, span = 'year', bounds = 'regular'):   
    for t in range(len(stockTickers)):
        companyName = str(rh.get_name_by_symbol(stockTickers[t]))
        stockHistogram = rh.stocks.get_historicals(stockTickers[t], span=span, bounds=bounds)
        stockHistogramDF = pd.DataFrame()
        for i in range(len(stockHistogram)):
            df = pd.DataFrame(stockHistogram[i], index = [i])
            stockHistogramDF = pd.concat([stockHistogramDF,df])
        stockHistogramDF.begins_at = pd.to_datetime(stockHistogramDF.begins_at, infer_datetime_format=True)
        stockHistogramDF.open_price = stockHistogramDF.open_price.astype('float32')
        stockHistogramDF.close_price = stockHistogramDF.close_price.astype('float32')
        stockHistogramDF.high_price = stockHistogramDF.high_price.astype('float32')
        stockHistogramDF.low_price = stockHistogramDF.low_price.astype('float32')
        ax = stockHistogramDF.plot(x = 'begins_at', y = 'open_price', figsize = (10,6))
        ax.fill_between(stockHistogramDF.begins_at, stockHistogramDF.low_price, stockHistogramDF.high_price, alpha=0.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend([stockTickers[t]])
        ax.set_title(companyName)
        plt.show()

#Function that extracts all stocks owned by user 
def extract_list ():
    stockTickerList = list(rh.build_holdings(True).keys())
    return stockTickerList

#Function to print all open positions
def print_open_positions():
    print("LIG: Printing Open Positions")
    time.sleep(2)
    print("LIG: Printing all Open Options Positions ")
    print(rh.options.get_open_option_positions())
    print("LIG: Printing all open Stock Positions")
    print(rh.account.get_open_stock_positions())

''''
Function that transforms the trading algorithm into real time context
Assumptions: 
    Bar Length for resampling is 15 seconds
    Average of Highs and Lows
'''
def simple_automated_trade (symbol,data, dataframe, amount, algorithm):
    global currentPosition
    dataFrameLength = len(dataframe)
    algorithm = algorithm
    resampledDataFrame = dataframe.resample('15s',label='right').last().ffill()
    resampledDataFrame["avg"] = resampledDataFrame[["high","low"]].mean(axis=1)
    resampledDataFrame["returns"] = np.log(resampledDataFrame["avg"] / resampledDataFrame["avg"].shift(1))
    resampledDataFrame["direction"] = np.where(resampledDataFrame["returns"] > 0, 1, -1)
    dataFeatures = resampledDataFrame["direction"]
    dataFeatures = dataFeatures.values.reshape(1, -1)
    tradeSignal = algorithm.predict(dataFeatures)[0]

    #Long the Stock
    if position in [0, -1] and signal == 1:
        #rh.orders.order_buy_fractional_by_price(symbol,amount)
        position = 1
        print("LIG: Going Long on {} for ${} at {}".format(symbol,amount,time.clock_gettime_ns(time.CLOCK_REALTIME)))
    
    #Short the stock 
    elif position in [0, 1] and signal == -1:
        #rh.orders.order_sell_fractional_by_price(symbol,amount)
        position = -1
        print("LIG: Going Short on {} for ${} at {}".format(symbol,amount,time.clock_gettime_ns(time.CLOCK_REALTIME)))

    else:
        print("LIG: No Trade Placed")
    

    #Closing the position 
    if len(resampledDataFrame) > 400:
        rh.orders.cancel_all_stock_orders()
        print("LIG: Closing out all open positions")





