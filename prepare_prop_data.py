# prepare data
import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
#import matplotlib

from scipy.stats import skew
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency

#import seaborn as sns
#import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import os
os.chdir('/Users/jingwang/Desktop/winter2019/si699')


# Put this when it's called
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

from sklearn import tree
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet




# replace id by popularity
def popularity_replace(data, feature):
    counts = data[feature].value_counts()
    d = {}
    for i in counts.index:
        d[i] = counts[i]
    feature_id = data[feature].tolist()
    feature_pop = []
    for i in feature_id:
        feature_pop.append(d[i])
    data[feature+'popularity'] = feature_pop
    return data
    
# get the specific prop data aggregate by day
def get_prop_id_day(data, id):
    comp_features = ['comp1_rate','comp1_inv','comp1_rate_percent_diff','comp2_inv','comp2_rate','comp2_rate_percent_diff','comp3_rate','comp3_inv','comp3_rate_percent_diff','comp4_rate','comp4_inv','comp4_rate_percent_diff','comp5_rate','comp5_inv','comp5_rate_percent_diff','comp6_rate','comp6_inv','comp6_rate_percent_diff','comp7_rate','comp7_inv','comp7_rate_percent_diff','comp8_rate','comp8_inv','comp8_rate_percent_diff']
    user_features = ['visitor_hist_starrating','visitor_hist_adr_usd','srch_query_affinity_score','orig_destination_distance','site_id','visitor_location_country_id','srch_id']
    other_features = ['random_bool']
    time_features = ['date_time']
    data = data.drop(columns = comp_features)
    data = data.drop(columns = user_features)
    data = data.drop(columns = other_features)
    data['date_time']=pd.to_datetime(data.date_time)
    data.sort_values(by=['date_time'], inplace = True)
    data = data.reset_index(drop=True)
    data = popularity_replace(data,'prop_country_id')
    data = popularity_replace(data,'srch_destination_id')
    data_new = data.drop(columns = 'prop_country_id')
    data_new = data_new.drop(columns = 'srch_destination_id')
    data_id = data_new[data_new['prop_id']==id]
    data_id = data_id.set_index('date_time')
    data_id_day = data_id.resample('D').median()
    data_id_day = data_id_day.drop(columns = 'prop_id')
    return data_id_day

# get the split data for cross validation
# by KFold, n_split = 5
class data_get():
    def __init__(self,data,variables):
        self.data = data.iloc[:]
        self.train_data = pd.DataFrame()
        #self.val_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        #self.model = model
        self.variables = variables
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        #self.X_val = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_test = pd.DataFrame()
        #self.y_val = pd.DataFrame()

        self.categorical_vars = ['srch_id','site_id','visitor_location_country_id','visitor_hist_starrating','prop_country_id','prop_id',
        'srch_destination_id']

        self.categorical_binary_vars = []
        self.continuous_vars = []

        self.X_train_normalized = pd.DataFrame()
        self.X_val_normalized = pd.DataFrame()
        self.X_test_normalized = pd.DataFrame()

        self.X_train_standardized = pd.DataFrame()
        self.X_val_standardized = pd.DataFrame()
        self.X_test_standardized = pd.DataFrame()
        self.X_train_kf = []
        self.X_test_kf = []
        self.y_train_kf = []
        self.y_test_kf = []
        self.X_train_ts = []
        self.X_test_ts = []
        self.y_train_ts = []
        self.y_test_ts = []
 
    def split_data(self):
        training_size = int(len(self.data) * 0.8)   
 
        test_size = int(len(self.data) * 0.2)
        print('training size: %d'%training_size)
      
        print('test size: %d'%test_size)
        # split data by temporal order
        self.train_data = self.data.iloc[0: training_size]
        #self.val_data = self.data.iloc[training_size:(training_size + validation_size)]
        # self.test_data = self.data.iloc[(training_size + validation_size): (training_size + validation_size + test_size)]
        self.test_data = self.data.iloc[training_size:]
        return self.train_data, self.test_data
    
    def get_X_y(self):
        # TODO: need to handle 'date_time' properly
        # for now, leave out "date_time" from modeling
        # self.variables += [col for col in self.data.columns.unique().tolist() if col not in ['price_usd','date_time']]
        self.X_train = self.train_data[self.variables]
        self.y_train = self.train_data['price_usd']
        self.X_test = self.test_data[self.variables]
        self.y_test = self.test_data['price_usd']
        return self.X_train, self.y_train, self.X_test, self.y_test 
    # KFold method to split data for cross validation
    def get_kf_list(self):
        kf = KFold(n_splits = 5)
        result = []
        for train, test in kf.split(self.X_train):
            result.append([train, test])
        n = 0
        for i in result:
            self.X_train_kf.append(self.X_train.iloc[result[n][0]])
            self.X_test_kf.append(self.X_train.iloc[result[n][1]])
            self.y_train_kf.append(self.y_train.iloc[result[n][0]])
            self.y_test_kf.append(self.y_train.iloc[result[n][1]])
            n =+ 1

        return self.X_train_kf, self.X_test_kf, self.y_train_kf, self.y_test_kf
# TimeSeries method to split data for cross validation
    def get_ts_list(self):
        tscv = TimeSeriesSplit(n_splits=5)
        result_ts = []
        for train, test in tscv.split(self.X_train):
            result_ts.append([train, test])
        n = 0
        for i in result_ts:
            self.X_train_ts.append(self.X_train.iloc[result_ts[n][0]])
            self.X_test_ts.append(self.X_train.iloc[result_ts[n][1]])
            self.y_train_ts.append(self.y_train.iloc[result_ts[n][0]])
            self.y_test_ts.append(self.y_train.iloc[result_ts[n][1]])
            n =+ 1
        return self.X_train_ts, self.X_test_ts, self.y_train_ts, self.y_test_ts

def get_RMSE(models,X_train_ts,X_test_ts,y_train_ts,y_test_ts):
    RMSE_test_lt = []
    RMSE_train_lt = []
    for i in range(5):
        reg = models
        reg.fit(X_train_ts[i],y_train_ts[i])
        pred_test = reg.predict(X_test_ts[i])
        RMSE_test = np.sqrt(mean_squared_error(y_test_ts[i],pred_test))
        RMSE_test_lt.append(RMSE_test)
        pred_train = reg.predict(X_train_ts[i])
        RMSE_train = np.sqrt(mean_squared_error(y_train_ts[i],pred_train))
        RMSE_train_lt.append(RMSE_train)
        #print(RMSE_train_lt, RMSE_test_lt)
    return RMSE_train_lt, RMSE_test_lt, pred_train, pred_test


train = pd.read_csv('./expediadata/train.csv')
test = pd.read_csv('./expediadata/test.csv')
cols_train_only = [col for col in train.columns.unique().tolist() if col not in test.columns.unique().tolist()]
train = train.drop(columns = cols_train_only)
all_data = pd.concat([train, test], ignore_index=True)
print('all_data is ready')

data_update = popularity_replace(all_data, 'prop_country_id')
data_update = popularity_replace(data_update, 'srch_destination_id')
data_day = get_prop_id_day(data_update, id = 116942)
print("data_day is ready")

variables = [col for col in data_day.columns.unique().tolist() if col not in ['price_usd']]
data_split = data_get(data_day,variables)
train_data, test_data = data_split.split_data()
X_train,y_train,X_test,y_test = data_split.get_X_y()
X_train_kf,X_test_kf,y_train_kf,y_test_kf = data_split.get_kf_list()
X_train_ts,X_test_ts,y_train_ts,y_test_ts = data_split.get_ts_list()

RMSE_train_ts, RMSE_test_ts, pred_train_ts, pred_test_ts = get_RMSE(ElasticNet(), X_train_ts,X_test_ts,y_train_ts,y_test_ts)
RMSE_train_kf, RMSE_test_kf, pred_train_kf, pred_test_kf = get_RMSE(ElasticNet(), X_train_kf,X_test_kf,y_train_kf,y_test_kf)

print("RMSE_train_ts:", RMSE_train_ts, "RMSE_test_ts:",RMSE_test_ts, 
    "RMSE_train_kf:", RMSE_train_kf,"RMSE_test_kf:",RMSE_test_kf)
