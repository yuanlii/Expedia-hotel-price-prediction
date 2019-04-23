import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import os
os.chdir('/Users/jingwang/Desktop/winter2019/si699/si699codes')

# Put this when it's called
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score

from sklearn import tree
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet

class modeling_pipeline():
    def __init__(self,data,model,variables):
        self.data = data.iloc[:]
        self.train_data = pd.DataFrame()
        self.val_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.model = model
        self.variables = variables
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.X_val = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.y_val = pd.DataFrame()

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
 
    def split_data(self):
        training_size_large = int(len(self.data) * 0.8)   
        validation_size = int(training_size_large * 0.2)
        training_size = training_size_large - validation_size
        test_size = int(len(self.data) * 0.2)
        print('training size: %d'%training_size)
        print('validation size: %d'%validation_size)
        print('test size: %d'%test_size)
        # split data by temporal order
        self.train_data = self.data.iloc[0: training_size]
        self.val_data = self.data.iloc[training_size:(training_size + validation_size)]
        # self.test_data = self.data.iloc[(training_size + validation_size): (training_size + validation_size + test_size)]
        self.test_data = self.data.iloc[(training_size + validation_size):]
        return self.train_data, self.val_data, self.test_data
    
    def divide_variables(self):
        # divide variables into categories
        # get categorical variables
        other_cols = [col for col in self.variables if col not in self.categorical_vars]
        # get categorical binary variables
        self.categorical_binary_vars += ['promotion_flag']
        self.categorical_binary_vars += [col for col in self.data if col.startswith('new')]
        self.categorical_binary_vars += [col for col in self.data if col.endswith('inv')]
        self.categorical_binary_vars += [col for col in self.data if col.endswith('bool')]
        # get continous variables
        self.continuous_vars += [ col for col in self.variables if (col not in self.categorical_binary_vars) & (col not in self.categorical_vars )]
        print ("categorical binary vars: ", len(self.categorical_binary_vars))
        print ("categorical non binary vars: ", len(self.categorical_vars))
        print ("continues vars: ", len(self.continuous_vars))
        return self.categorical_vars, self.categorical_binary_vars, self.continuous_vars

    def get_X_y(self):
        # TODO: need to handle 'date_time' properly
        # for now, leave out "date_time" from modeling
        # self.variables += [col for col in self.data.columns.unique().tolist() if col not in ['price_usd','date_time']]
        self.X_train = self.train_data[self.variables]
        self.y_train = self.train_data['price_usd']
        self.X_val = self.val_data[self.variables]
        self.y_val = self.val_data['price_usd']
        self.X_test = self.test_data[self.variables]
        self.y_test = self.test_data['price_usd']
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test

    def get_normalized_X_y(self):
        normalizer = Normalizer().fit(self.X_train) 
        self.X_train_normalized = normalizer.transform(self.X_train)
        self.X_val_normalized = normalizer.transform(self.X_val)
        self.X_test_normalized = normalizer.transform(self.X_test)
        return self.X_train_normalized, self.X_val_normalized, self.X_test_normalized
    
    def get_standardized_X_y(self):
        # usign min-max scaler to standardized
        scaler = MinMaxScaler().fit(self.X_train)
        self.X_train_standardized = scaler.transform(self.X_train)
        self.X_val_standardized = scaler.transform(self.X_val)
        return self.X_train_standardized, self.X_val_standardized, self.X_test_standardized 

    def get_RMSE(self,y_pred,y_true,data):
        return np.sqrt(sum((y_pred - y_true)**2)/len(data))
    
    # updated: adding y_pred_test and test_RMSE
    def get_modeling_result(self):
        reg = self.model.fit(self.X_train, self.y_train)
        y_pred_val = reg.predict(self.X_val)
        y_pred_train = reg.predict(self.X_train)
        y_pred_test = reg.predict(self.X_test)
        val_RMSE = self.get_RMSE(y_pred_val, self.y_val, self.val_data)
        train_RMSE = self.get_RMSE(y_pred_train ,self.y_train, self.train_data)
        test_RMSE = self.get_RMSE(y_pred_test ,self.y_test, self.test_data)
        print('training RMSE:',train_RMSE)
        print('valiation RMSE:',val_RMSE)
        print('test RMSE:',test_RMSE)
        return y_pred_train, y_pred_val, y_pred_test

    def get_normalized_modeling_result(self):
        reg = self.model.fit(self.X_train_normalized, self.y_train)
        y_pred_val = reg.predict(self.X_val_normalized)
        y_pred_train = reg.predict(self.X_train_normalized)
        val_RMSE = self.get_RMSE(y_pred_val, self.y_val, self.val_data)
        train_RMSE = self.get_RMSE(y_pred_train ,self.y_train, self.train_data)
        print('training RMSE:',train_RMSE)
        print('valiation RMSE:',val_RMSE)
        return train_RMSE, val_RMSE

    def get_standardized_modeling_result(self):
        reg = self.model.fit(self.X_train_standardized, self.y_train)
        y_pred_val = reg.predict(self.X_val_standardized)
        y_pred_train = reg.predict(self.X_train_standardized)
        val_RMSE = self.get_RMSE(y_pred_val, self.y_val, self.val_data)
        train_RMSE = self.get_RMSE(y_pred_train ,self.y_train, self.train_data)
        print('training RMSE:',train_RMSE)
        print('valiation RMSE:',val_RMSE)
        return train_RMSE, val_RMSE

class PropModeling():
    def __init__(self):
        pass
    
    def load_data(self):
        train = pd.read_csv('../expediadata/train.csv')
        test = pd.read_csv('../expediadata/test.csv')
        cols_train_only = [col for col in train.columns.unique().tolist() if col not in test.columns.unique().tolist()]
        train = train.drop(columns = cols_train_only)
        all_data = pd.concat([train, test], ignore_index=True)
        return all_data
    
    def popularity_id(self,all_data,feature):
        id_counts = all_data[feature].value_counts()
        d = {}
        for ID in id_counts.index:
            d[ID] = id_counts[ID]
        feature_id = all_data[feature].tolist()
        popularity = []
        for ID in feature_id:
            popularity.append(d[ID])
        return popularity 
    
    def prop_modeling(self, all_data, prop_id):
        # get features
        comp_features = ['comp1_rate','comp1_inv','comp1_rate_percent_diff','comp2_inv','comp2_rate','comp2_rate_percent_diff','comp3_rate','comp3_inv','comp3_rate_percent_diff','comp4_rate','comp4_inv','comp4_rate_percent_diff','comp5_rate','comp5_inv','comp5_rate_percent_diff','comp6_rate','comp6_inv','comp6_rate_percent_diff','comp7_rate','comp7_inv','comp7_rate_percent_diff','comp8_rate','comp8_inv','comp8_rate_percent_diff']
        user_features = ['visitor_hist_starrating','visitor_hist_adr_usd','srch_query_affinity_score','orig_destination_distance','site_id','visitor_location_country_id','srch_id']
        other_features = ['random_bool']
        time_features = ['date_time']
        all_data = all_data.drop(columns = comp_features)
        all_data = all_data.drop(columns = user_features)
        all_data = all_data.drop(columns = other_features)
        all_data['date_time'] = pd.to_datetime(all_data.date_time)
        all_data.sort_values(by=['date_time'],inplace=True)
        all_data = all_data.reset_index(drop=True)
        
        # change id to popularity
        
        # handle country
        country_counts = all_data['prop_country_id'].value_counts()

        d = {}
        for ID in country_counts.index:
            d[ID] = country_counts[ID]
        country_id = all_data['prop_country_id'].tolist()
        country_pop = []
        for ID in country_id:
            country_pop.append(d[ID])
        all_data['country_value_counts'] = country_pop
        city_counts = all_data['srch_destination_id'].value_counts()

        # handle city
        city = {}
        for ID in city_counts.index:
            city[ID] = city_counts[ID]
        city_id = all_data['srch_destination_id'].tolist()
        city_pop = []
        for ID in city_id:
            city_pop.append(city[ID])
        all_data['city_value_counts'] = city_pop

        all_data_new = all_data.drop(columns = 'prop_country_id')
        all_data_new = all_data_new.drop(columns = 'srch_destination_id')
        prop = all_data_new[all_data_new['prop_id']==prop_id]

        all_data_t = prop.set_index('date_time')
        prop_day = all_data_t.resample('D').median()
        prop_day.count()

        prop_day = prop_day.drop(columns = 'prop_id')

        
        variables = [col for col in prop_day.columns.unique().tolist() if col not in ['price_usd']]
        model_prop_e = modeling_pipeline(prop_day, ElasticNet(), variables)
        train_data, val_data, test_data = model_prop_e.split_data()
        X_train,y_train,X_val,y_val,X_test,y_test = model_prop_e.get_X_y()
        y_pred_train, y_pred_val, y_pred_test = model_prop_e.get_modeling_result()
        
        return y_pred_train, y_pred_val, y_pred_test


# p = PropModeling()
# p_data = p.load_data()
# y_pred_train, y_pred_val, y_pred_test=p.prop_modeling(p_data, 116942)
# print('y_pred_train:',y_pred_train,'y_pred_val:',y_pred_val,'y_pred_test',y_pred_test)

        