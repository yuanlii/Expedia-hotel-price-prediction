import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


class Reformat_data():
    def __init__(self):
        self.data = pd.DataFrame()
        self.variables = []
        self.categorical_vars = []
        self.categorical_binary_vars = []
        self.continuous_vars = []
        self.dest_data_list = {}
        # each property and its associated data
        self.prop_data_list = {}
        self.datetime_range = []
        # list destination id with enough data
        self.valid_dest_list = []
        self.daily_price_data = []
        # self.all_daily_price = pd.DataFrame()
        self.all_daily_price = {}
    
    def load_data(self, data_file_path):
        self.data = pd.read_csv(data_file_path, encoding = 'utf-8')
        return self.data
        
    def divide_variables(self):
        self.variables += [col for col in self.data.columns.unique().tolist() if col not in ['price_usd','date_time']]
        self.categorical_vars += ['srch_id','site_id','visitor_location_country_id','visitor_hist_starrating','prop_country_id','prop_id','prop_starrating',
        'srch_destination_id']
        other_cols = [col for col in self.variables if col not in self.categorical_vars]
        # get categorical binary variables
        self.categorical_binary_vars += ['promotion_flag']
        self.categorical_binary_vars += [col for col in self.data if col.startswith('new')]
        self.categorical_binary_vars += [col for col in self.data if col.endswith('inv')]
        self.categorical_binary_vars += [col for col in self.data if col.endswith('bool')]
        # get continous variables
        self.continuous_vars += [ col for col in self.variables if (col not in self.categorical_binary_vars) & (col not in self.categorical_vars )]
        return self.categorical_vars,self.categorical_binary_vars,self.continuous_vars
    
    def get_data_by_dest(self):
        '''separate entire dataset by destinations; append to a list'''
        srch_destination_ids = self.data['srch_destination_id'].unique().tolist()
        for srch_destination_id in srch_destination_ids:
            destination_data = self.data[self.data['srch_destination_id'] == srch_destination_id]
            self.dest_data_list[srch_destination_id] = destination_data
        return self.dest_data_list

    # updated: get data by property_id
    def get_data_by_prop(self):
        prop_ids = self.data['prop_id'].unique().tolist()
        for prop_id in prop_ids:
            prop_data = self.data[self.data['prop_id'] == prop_id]
            self.prop_data_list[prop_id] = prop_data
        return self.prop_data_list

    
    def get_datetime_range(self):
        # covert 'date_time' to datetime object
        self.data['date_time'] = pd.to_datetime(self.data.date_time)
        # resample by day
        data = self.data.set_index('date_time')
        price_data = data['price_usd'].resample('D').median()
        self.datetime_range += price_data.index.tolist()
        return self.datetime_range
    
    def get_daily_price(self, dest_id):
        dest_data = self.dest_data_list[dest_id]
        dest_data['date_time'] = pd.to_datetime(dest_data.date_time)
        dest_data = dest_data.set_index('date_time')
        dest_daily_price = dest_data['price_usd'].resample('D').median()
        return dest_daily_price
    
    def get_valid_dests(self):
        '''get a list of destination ids covering at least 50% of the datetime range'''
        for dest_id in self.dest_data_list.keys():
            # after remove na, if num of available records exceed 50% percentile of all date range, then keep it; otherwise, remove destination from dataset
            if len(self.get_daily_price(dest_id).dropna()) >= len(self.datetime_range)*0.5:
                self.valid_dest_list.append(dest_id)
        return self.valid_dest_list
    
    
    def get_all_daily_price(self):
        '''get a dictionary --> key:destination_id, value: daily price dataframe'''
        for dest_id in self.dest_data_list.keys():
            self.all_daily_price[dest_id] = self.get_daily_price(dest_id)
        return self.all_daily_price

    def resample_price_by_day(self):
        time_related = ['date_time','price_usd']
        time_data = self.data[time_related]
        # resample data by day
        time_data['date_time'] = pd.to_datetime(time_data.date_time)
        time_data = time_data.set_index('date_time')
        daily_price = time_data['price_usd'].resample('D').median()
        return daily_price

    def get_month_signal(self):
        pass
                