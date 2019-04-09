import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import os
# os.chdir('/Users/liyuan/desktop/SI699/codes')
from scipy import stats
from sklearn.linear_model import LinearRegression

from reformat_data_by_day import Reformat_data
from model_pipeline import modeling_pipeline

# by property raise the sparsity problem --> can try aggregate by region + by property or by region + by star-rating

class TimeFeatureModeling():
    def __init__(self):
        self.data = pd.Series()
        self.prop_counts = {}
        self.prop_ranked = []
        self.price_by_prop = pd.Series()

    def load_data(self, input_data_path):
        rf = Reformat_data()
        self.data = rf.load_data(input_data_path)
        self.data['date_time'] = pd.to_datetime(self.data['date_time'])
        return self.data
    
    def get_counts_by_prop(self):
        '''get a dictionary listing the prop_id as key and the number of records in data accordingly as value'''

        print('there are %d unique properties'%len(self.data.prop_id.unique().tolist()))
        prop_ids = self.data.prop_id.unique().tolist()
        for prop_id in prop_ids:
            self.prop_counts[prop_id] = len(self.data[self.data['prop_id']== prop_id])
        # sort prop_id by descending order based on its counts in data
        self.prop_ranked = sorted(self.prop_counts.keys(), key = lambda x: self.prop_counts[x], reverse = True)
        return self.prop_ranked


    def agg_by_prop(self):
        # format data by property --> we need to predict price for each property
        self.price_by_prop = self.data[['date_time','prop_id','price_usd']]
        self.price_by_prop = self.price_by_prop[['prop_id','price_usd']].groupby('prop_id').mean()
        return self.price_by_prop


    # can pass in data per property
    def extract_time_features(self,data):
        '''takes in dataframe with all fields and output dataframe with only price and date time info'''
        price_data = data[['date_time','price_usd']]
        # extract "day","month", "quarter" signals and create new columns
        price_data['day'] = price_data['date_time'].apply(lambda x: x.day)
        price_data['week'] = price_data['date_time'].apply(lambda x: x.week)
        price_data['month'] = price_data['date_time'].apply(lambda x: x.month)
        price_data['quarter'] = price_data['date_time'].apply(lambda x: x.quarter)
        print('price data len:', len(price_data)) # debugging
        return price_data

    # we would want to observe the variations of price data
    def plot_data_trend(self,price_data):
        # price_data = data[['date_time','price_usd']]
        price_data = price_data.set_index('date_time')
        daily_price = price_data['price_usd'].resample('D').median()
        print('variance of data: %d'%np.var(daily_price))
        sns.lineplot(x = daily_price.index, y = daily_price)
        # daily_price.plot()
    
    # hope to observe variations by day, month, quarter, etc. --> so such features can be used as indicators for prediction
    def agg_by_month(self, price_data):
        # using mean as comparison metric
        price_by_month = price_data[['price_usd','month']].groupby('month').mean()
        print('variance of data: %d'%np.var(price_by_month))
        return price_by_month

    def agg_by_day(self, price_data):
        # group by day
        price_by_day = price_data[['price_usd','day']].groupby('day').mean()
        print('variance of data: %d'%np.var(price_by_day))
        return price_by_day
    
    def agg_by_quarter(self, price_data):
        # group by quarter
        price_by_quarter = price_data[['price_usd','quarter']].groupby('quarter').mean()
        print('variance of data: %d'%np.var(price_by_quarter))
        return price_by_quarter
    
    def agg_by_week(self, price_data):
        # group by week
        price_by_week = price_data[['price_usd','week']].groupby('week').mean()
        print('variance of data: %d'%np.var(price_by_week))
        return price_by_week
        
    # with time features that we extracted, it can be used to build regression model
    def regression_model(self, price_data):
        # update: adding 'week'
        variables = ['day','month','quarter','week']
        mp = modeling_pipeline(price_data,LinearRegression(),variables)
        mp.split_data()
        X_train, y_train, X_val, y_val, X_test, y_test = mp.get_X_y()
        y_pred_train, y_pred_val, y_pred_test = mp.get_modeling_result()
        return y_pred_train, y_pred_val, y_pred_test
    
    def regression_model_updated(self,):
        pass


    
    

