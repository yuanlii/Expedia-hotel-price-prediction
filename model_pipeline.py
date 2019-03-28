import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

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
    
    def get_modeling_result(self):
        reg = self.model.fit(self.X_train, self.y_train)
        y_pred_val = reg.predict(self.X_val)
        y_pred_train = reg.predict(self.X_train)
        val_RMSE = self.get_RMSE(y_pred_val, self.y_val, self.val_data)
        train_RMSE = self.get_RMSE(y_pred_train ,self.y_train, self.train_data)
        print('training RMSE:',train_RMSE)
        print('valiation RMSE:',val_RMSE)
        return train_RMSE, val_RMSE

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