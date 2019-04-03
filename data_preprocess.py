import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error

class Data_preprocess():
    def __init__(self, train_file_path, test_file_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.all_data = pd.DataFrame()
        self.sampled_data = pd.DataFrame()
    
    def load_data(self):
        train = pd.read_csv(self.train_file_path)
        test = pd.read_csv(self.test_file_path)
        print('training data has %d records'%len(train))
        print('test data has %d records'%len(test))
        
        # drop columns in training data that are not available in test data set, including :'position', 'click_bool', 'gross_bookings_usd', 'booking_bool'
        cols_train_only = [col for col in train.columns.unique().tolist() if col not in test.columns.unique().tolist()]
        print('Columns only available in training data:',cols_train_only)
        train = train.drop(columns = cols_train_only)

        # combine train and test data
        self.all_data = pd.concat([train, test], ignore_index=True)
        print('Whole dataset has %d records' % len(self.all_data))
        
        # convert 'date_time' to datatime object
        self.all_data['date_time'] = pd.to_datetime(self.all_data.date_time)
        self.all_data.sort_values(by=['date_time'],inplace=True)
        self.all_data = self.all_data.reset_index(drop=True)
        return self.all_data    
    
    def clean_data(self,data, output_file_name):
        # handle NA values
        NA_columns = []
        for col in data.columns.unique().tolist():
            if data[col].isna().values.any() == True:
                NA_columns.append(col)
        for col in NA_columns:
            # create binary columns
            new_col = 'new_'+ col
            data[new_col] = data[col].apply(lambda x: 1 if x >= 0 else 0)
        # replace old column NA values to median value
        data = data.fillna(data.median())
        # output to csv file
        data.to_csv(output_file_name +'.csv',index = False, encoding = 'utf-8')
        return data
    
    def sample_data(self,sample_size):
        interval_range = len(self.all_data)//sample_size
        mid_idx_lst = []
        for i in range(1,sample_size+1):
            mid_idx = (interval_range*(i-1) + interval_range*i)//2
            mid_idx_lst.append(mid_idx)
        self.sampled_data = self.all_data.iloc[mid_idx_lst]
        return self.sampled_data

    