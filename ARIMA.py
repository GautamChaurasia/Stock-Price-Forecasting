import os
from datetime import datetime as dt
import yfinance as yf
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',FutureWarning)

class Predict:

    def __init__(self, stock, st:str = '1972-6-1', e:str = dt.now().strftime('%Y-%m-%d')):
        self.stk = stock
        self.start = st
        self.end = e

    def get_data(self):
        stock = yf.Ticker(self.stk)
        df = stock.history(start=self.start, end=self.end, actions=False)
        df_close = df['Close']
        return df_close

    def visualize_data(self, df_close):
        fontd={'size':13}
        sns.set_style('whitegrid')
        plt.figure(figsize=(15,8))
        plt.plot(df_close)
        plt.title(self.stk, fontdict={'style':'oblique','color':'blue', 'size':15})
        plt.xlabel('Year', fontdict=fontd)
        plt.ylabel('Closing Price', fontdict=fontd)
        plt.gcf().autofmt_xdate()
        
        if os.path.exists('static/images/'+self.stk+''):
            if os.path.isfile('static/images/'+self.stk+'/'+self.stk+'_close.png'):
                os.remove('static/images/'+self.stk+'/'+self.stk+'_close.png')
            plt.savefig('static/images/'+self.stk+'/'+self.stk+'_close.png')
        else:
            os.mkdir('static/images/'+self.stk+'')
            plt.savefig('static/images/'+self.stk+'/'+self.stk+'_close.png')
        return 'static/images/'+self.stk+'/'+self.stk+'_close.png'

    def stationarize_data(self, timeseries):
        # rolmean = timeseries.rolling(12).mean()
        # rolstd = timeseries.rolling(12).std()
        adft = adfuller(timeseries,autolag='AIC')
        # print("Results of dickey fuller test")
        # output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
        if adft[1]>0.05:
            self.df_log = np.log(timeseries)
        else: self.df_log = timeseries
        print('Data Stationarized!')

    def forecast(self, period = '1m'):
        self.sel_tf = {'1w':(3,7), '1m':(5,30), '1y':(7,365)}
        self.per = period
        self.no_fc = self.sel_tf[self.per][1]
        self.train_data = self.df_log[:len(self.df_log)]
        self.datelist = pd.date_range(dt.now().strftime('%Y-%m-%d'), periods=self.no_fc).tolist()

        model_autoARIMA = auto_arima(self.train_data[len(self.train_data)-(self.sel_tf[self.per][0]*self.no_fc):],
                            start_p=0, start_q=0,
                            test='adf',       # use adftest to find optimal 'd'
                            max_p=3, max_q=3, # maximum p and q
                            m=1,              # frequency of series
                            d=None,           # let model determine 'd'
                            seasonal=False,   # No Seasonality
                            start_P=0, 
                            D=0, 
                            trace=False,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True)
        ord = tuple(map(int, str(model_autoARIMA)[7:12].split(',')))
        # print(ord)

        model = ARIMA(self.train_data[len(self.train_data)-(self.sel_tf[self.per][0]*self.no_fc):], order=ord)  
        fitted = model.fit(disp=-1)  
        # print(fitted.summary())

        # Forecast
        self.fc, se, conf = fitted.forecast(len(self.datelist), alpha=0.05)  # 95% conf

        # print(pd.DataFrame({'Actual':self.test_data, 'Predicted':self.fc}))

        # Make as pandas series
        self.fc_series = pd.Series(self.fc, index=self.datelist)
        self.lower_series = pd.Series(conf[:, 0], index=self.datelist)
        self.upper_series = pd.Series(conf[:, 1], index=self.datelist)
        return np.exp(self.fc), self.datelist

    def save_fig(self):
        zoom_out_fact = self.sel_tf[self.per][0]
        # Plot
        plt.figure(figsize=(10,5), dpi=100)
        plt.plot(np.exp(self.train_data[len(self.train_data)-(zoom_out_fact*self.no_fc):]), label='training data')
        # plt.plot(np.exp(self.test_data), color = 'blue', label='Actual Stock Price')
        plt.plot(np.exp(self.fc_series), color = 'orange',label='Forecasted Stock Price')
        plt.fill_between(self.lower_series.index, np.exp(self.lower_series), np.exp(self.upper_series), 
                        color='k', alpha=.10)
        plt.title(self.stk, fontdict={'style':'oblique','color':'blue', 'size':15})
        plt.xlabel('Year')
        plt.ylabel('Closing Price')
        plt.legend(loc='upper left', fontsize=8)
        plt.gcf().autofmt_xdate()
        if os.path.exists('static/images/'+self.stk+''):
            if os.path.isfile('static/images/'+self.stk+'/'+self.stk+'.png'):
                os.remove('static/images/'+self.stk+'/'+self.stk+'.png')
            plt.savefig('static/images/'+self.stk+'/'+self.stk+'.png')
        else:
            os.mkdir('static/images/'+self.stk+'')
            plt.savefig('static/images/'+self.stk+'/'+self.stk+'.png')
        return 'static/images/'+self.stk+'/'+self.stk+'.png'

    def check_performance(self):
        pfmc = dict()
        # report performance
        mse = mean_squared_error(self.test_data, self.fc)
        pfmc['mse'] = mse
        mae = mean_absolute_error(self.test_data, self.fc)
        pfmc['mae'] = mae
        rmse = math.sqrt(mean_squared_error(self.test_data, self.fc))
        pfmc['rmse'] = rmse
        mape = np.mean(np.abs(self.fc - self.test_data)/np.abs(self.test_data))
        pfmc['mape'] = mape
        return pfmcs