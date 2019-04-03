from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf

class Moving_average(object):
    def __init__(self, timeseries):
        self.timeseries = timeseries
        self.timeseries_lt = pd.Series()
        self.moving_avg = pd.Series()
        self.ts_diff = pd.Series()

    def test_stationarity(self, timeseries, sliding_window = 3):
        '''
        timeseries are pandas dataframe that has already convert datetime varible into "datetime" object, and format df using datetime as index
        '''
        #Determing rolling statistics
        rolmean = timeseries.rolling(sliding_window).mean()  
        rolstd = timeseries.rolling(sliding_window).std()
        #Plot rolling statistics:
        orig = plt.plot(timeseries, color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)
        
        #Perform Dickey-Fuller test:
        print ('Results of Dickey-Fuller Test:')
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print (dfoutput)
    
    def log_transform_timeseries(self):
        # log transform data
        self.timeseries_lt = np.log(self.timeseries)  
        plt.plot(self.timeseries_lt)
        plt.title('log transformed data trend')
        return self.timeseries_lt

    def plot_moving_average(self, timeseries, sliding_window=3):
        self.moving_avg = timeseries.rolling(sliding_window).mean()
        plt.plot(timeseries)
        plt.plot(self.moving_avg, color='red')
        plt.title('Data trend with moving average')

    def test_moving_avg_diff(self, timeseries):
        ts_moving_avg_diff = timeseries - self.moving_avg
        ts_moving_avg_diff.dropna(inplace=True)
        self.test_stationarity(ts_moving_avg_diff)

    def expweighted_moving_avg(self, timeseries, sliding_window = 3):
        # 指数加权移动平均法
        expwighted_avg = timeseries.ewm(sliding_window).mean() 
        plt.plot(timeseries)
        plt.plot(expwighted_avg, color='red')
        plt.title('expweighted_avg')
    
    def remove_seasonality(self, timeseries):
        # 差分 -> 消除季节性和趋势
        self.ts_diff = timeseries - timeseries.shift()
        plt.plot(self.ts_diff)
        self.ts_diff.dropna(inplace=True)
        self.test_stationarity(self.ts_diff)

    def test_autocorrelation(self, nlags = 3, method = 'ols'):
        lag_acf = acf(self.ts_diff, nlags)
        lag_pacf = pacf(self.ts_diff, nlags, method)

        # Plot ACF:  
        # 自相关函数（ACF）：这是时间序列和它自身滞后版本之间的相关性的测试
        plt.plot(lag_acf)
        plt.axhline(y=0,linestyle='--',color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(self.ts_diff)),linestyle='--',color='gray')
        plt.axhline(y=1.96/np.sqrt(len(self.ts_diff)),linestyle='--',color='gray')
        plt.title('Autocorrelation Function')
        print('another way of plotting ACF ...')
        plot_acf(self.timeseries_lt)
        plt.show()

        # Plot PACF:  
        # 部分自相关函数(PACF):这是时间序列和它自身滞后版本之间的相关性测试，但是是在预测（已经通过比较干预得到解释）的变量后。
        plt.plot(lag_pacf)
        plt.axhline(y=0,linestyle='--',color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(self.ts_diff)),linestyle='--',color='gray')
        plt.axhline(y=1.96/np.sqrt(len(self.ts_diff)),linestyle='--',color='gray')
        plt.title('Partial Autocorrelation Function')
        plt.tight_layout()

    def fit_AR_model(self,step = 2):
        # model = ARIMA(ts_stress, order=(2, 1, 0)) 
        model = ARIMA(self.timeseries_lt, order= (step, 1, 0))      
        results_AR = model.fit(disp=-1)
        plt.plot(self.ts_diff)
        plt.plot(results_AR.fittedvalues, color='red')
        plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-self.ts_diff)**2))
        # AR 倒回到原始区间 
        predictions_AR_diff = pd.Series(results_AR.fittedvalues, copy=True) 
        # AR 累计总和
        predictions_AR_diff_cumsum = predictions_AR_diff.cumsum()
        ### AR结果
        predictions_AR = pd.Series(self.timeseries_lt.ix[0], index=self.timeseries_lt.index)
        predictions_AR = predictions_AR.add(predictions_AR_diff_cumsum,fill_value=0)
        return predictions_AR

    def fit_MA_model(self, sliding_window = 3):
        # 移动平均数（MA）模型 
        model = ARIMA(self.timeseries_lt, order=(0, 1, sliding_window))   
        results_MA = model.fit(disp=-1)  
        plt.plot(self.ts_diff)
        plt.plot(results_MA.fittedvalues, color='red')
        plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-self.ts_diff)**2))
        # MA 倒回到原始区间 
        predictions_MA_diff = pd.Series(results_MA.fittedvalues, copy=True)
        # MA 累计总和
        predictions_MA_diff_cumsum = predictions_MA_diff.cumsum()
        ### MA结果
        predictions_MA = pd.Series(self.timeseries_lt.ix[0], index=self.timeseries_lt.index)
        predictions_MA = predictions_MA.add(predictions_MA_diff_cumsum,fill_value=0)
        return predictions_MA

    def fit_ARIMA_model(self,step = 2, sliding_window = 2):
        # 组合模型
        model = ARIMA(self.timeseries_lt, order=(step, 1, sliding_window))
        results_ARIMA = model.fit(disp=-1)  
        plt.plot(self.ts_diff)
        plt.plot(results_ARIMA.fittedvalues, color='red')
        plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-self.ts_diff)**2))
        # ARIMA(组合模型) 倒回到原始区间  
        predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
        # ARIMA 累计总和
        predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
        ### ARIMA结果（log transform） 
        predictions_ARIMA = pd.Series(self.timeseries_lt.ix[0], index=self.timeseries_lt.index)
        predictions_ARIMA = predictions_ARIMA.add(predictions_ARIMA_diff_cumsum,fill_value=0)
        return predictions_ARIMA
        
    def get_prediction(self,pred_model):
        '''pred_model is the log-transformed data; this function would convert the log-transformed data back to actual number'''
        pred_model_convert = np.exp(pred_model)
        plt.plot(self.timeseries)
        plt.plot(pred_model_convert)
        # plt.title('RMSE: %.4f'% np.sqrt(sum((pred_model_convert-self.timeseries)**2)/len(self.timeseries)))
        plt.title('RMSE: %.4f'% np.sqrt(sum((pred_model_convert-self.timeseries.shift())**2)/len(self.timeseries)))
        return pred_model_convert


  



        



                