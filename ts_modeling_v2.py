import pandas as pd
from pandas import datetime
import numpy as np
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from reformat_data_by_day import Reformat_data
from time_feature_modeling import TimeFeatureModeling
from model_pipeline import modeling_pipeline

class TsModeling(object):
    def __init__(self):
        self.data = pd.Series()
        self.daily_price = pd.Series()
        # self.training_size = 0
        # self.validation_size = 0
        # self.test_size = 0
    
        self.train = np.array([])
        self.val = np.array([])
        self.test = np.array([])
        
        # predictions
        self.ARIMA_val_predictions = np.array([])
        self.ARIMA_test_predictions = np.array([])
        self.regression_train_predictions = np.array([])
        self.regression_val_predictions = np.array([])
        self.regression_test_predictions = np.array([])


    def load_data(self, input_data_path):
        # load sampled data (10000 exampless)
        self.data = pd.read_csv('../res/cleaned_sampled_data_10000.csv')
        # resample data by day
        self.data['date_time'] = pd.to_datetime(self.data.date_time)
        self.data = self.data.set_index('date_time')
        self.daily_price = self.data['price_usd'].resample('D').median()
        return self.daily_price


    # get timeseries data for one destination group by starrating
    def process_data_by_dest(self,dest_data):
        dest_data['date_time'] = pd.to_datetime(dest_data.date_time)
        dest_data = dest_data.set_index('date_time')
        dest_data = dest_data[['prop_starrating','prop_id','price_usd']]
        return dest_data   


    def process_data_by_dest_by_starrating(self, dest_data): 
        ''' get the data for one destination and get the data related to one starrating, which has the largest number of records '''
        dest_data_by_starrating = {}
        dest_data = self.process_data_by_dest(dest_data)
        starrating_lst = dest_data['prop_starrating'].unique().tolist()
        for starrating in starrating_lst:
            dest_data_by_starrating[starrating] = dest_data[dest_data['prop_starrating'] == starrating]
        # get the starrating in this destination that are with the largest number of records
        sorted_starratings = sorted(dest_data_by_starrating, key = lambda x: len(dest_data_by_starrating[x]),reverse = True )
        dest_data_star = dest_data[dest_data['prop_starrating'] == sorted_starratings[0]]
        return dest_data_star


    def get_auto_correlation(self,data):
        ''' plot auto-correlation '''
        autocorrelation_plot(data)
        pyplot.show()


    def split_data(self, data):
        ''' split price data '''
        X = data.values
        training_size = int(len(X) * 0.8) 
        self.validation_size = int(training_size * 0.2)
        self.training_size = training_size - self.validation_size
        self.test_size = int(len(X) * 0.2)
        print('training size: %d'%self.training_size)
        print('validation size: %d'%self.validation_size)
        print('test size: %d'%self.test_size)
        # split data by temporal order
        self.train = X[0: self.training_size]
        self.val = X[self.training_size:(self.training_size + self.validation_size)]
        self.test = X[(self.training_size + self.validation_size):]
        return self.train, self.val, self.test


    def get_model_performance(self, history, data, p,d,q):
        '''[data] is source data what model is trying to predict on; 
            combined to use with fit_ARIMA_model '''
        history = [x for x in self.train]
        predictions = list()
        for t in range(len(data)):
            model = ARIMA(history, order=(p,d,q))  
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = data[t]
            history.append(obs)
        mse = mean_squared_error(data, predictions)
        rmse = np.sqrt(mse)
        print('RMSE: %.3f' % rmse)
        # plot
        pyplot.plot(data)
        pyplot.plot(predictions, color='red')
        pyplot.show()
        return predictions


    def fit_ARIMA_model(self, data, p,d,q):
        '''fit values and get predictions; data --> price data with date_time as index, format as a dataframe'''
        # split data
        self.train, self.val, self.test = self.split_data(data)
        # get training (validation) predictions
        history = [x for x in self.train]
        self.ARIMA_val_predictions  = self.get_model_performance(history,self.val,p,d,q)
        # get test predictions
        # concatenate train and val (np.array)
        concat = np.concatenate((self.train, self.val), axis=0)
        history = [x for x in concat]
        self.ARIMA_test_predictions = self.get_model_performance(history,self.test,p,d,q)
        return self.ARIMA_val_predictions, self.ARIMA_test_predictions


    def fit_regression_model(self, data):
        '''fit regression model'''
        tm = TimeFeatureModeling()
        price_data = tm.extract_time_features(data)
        tm.plot_data_trend(price_data)
        self.regression_y_pred_train, self.regression_y_pred_val, self.regression_y_pred_test = tm.regression_model(price_data)
        return self.regression_y_pred_train, self.regression_y_pred_val, self.regression_y_pred_test


    def get_mse(self, pred, true):
        '''pred -> prediction values, true -> true values '''
        mse = mean_squared_error(true, pred)
        return mse

    def get_rmse(self, pred, true):
        mse = mean_squared_error(true, pred)
        rmse = np.sqrt(mse)
        return rmse


    def second_data_prepare(self):
        '''valid_data e.g., hotel price by 1 dest by 1 starrating '''
        # regression results reformat
        # self.regression_y_pred_val[1:] --> handle dimension mismatch
        regression_y_pred_val = self.regression_y_pred_val[1:].reshape(-1,1)
        # regression_y_pred_val = self.regression_y_pred_val[:-1].reshape(-1,1)
        regression_y_pred_test = self.regression_y_pred_test.reshape(-1,1)
        print(regression_y_pred_val.shape)
        print(regression_y_pred_test.shape)

        # ARIMA results reformat
        ARIMA_val_predictions = np.array(self.ARIMA_val_predictions).reshape(-1,1)
        ARIMA_test_predictions = np.array(self.ARIMA_test_predictions).reshape(-1,1)
        print(ARIMA_val_predictions.shape)
        print(ARIMA_test_predictions.shape)

        # get y_train, y_test for 2nd layer modeling
        y_train = self.val
        y_test = self.test
        print('y_train shape:',y_train.shape)
        print('y_test shape:',y_test.shape)

        # concatenate predictions from two models; used as new input data for second-layer model
        X_train = np.concatenate(( regression_y_pred_val, ARIMA_val_predictions), axis=1)
        X_test = np.concatenate(( regression_y_pred_test, ARIMA_test_predictions), axis=1)
        return X_train, X_test, y_train, y_test


    # Notice: for second_layer_modeling, we would use validation prediction and test prediction after concatenation; in particular, using validation predictions as the new X_train for second-layer modeling, this is because in timeseries prediction, we can only feed into train data to fit ARIMA model, and predict afterwards (we cannot use train data to predict itself in ARIMA ts prediction model --> so we would use validation pred and test pred as training data for 2nd layer model)

    def second_layer_modeling(self, model):
        X_train, X_test, y_train, y_test = self.second_data_prepare()

        reg = model.fit(X_train, y_train)
        y_pred_train = reg.predict(X_train)
        y_pred_test = reg.predict(X_test)
        # get rmse
        train_RMSE = self.get_rmse(y_pred_train ,y_train)
        test_RMSE = self.get_rmse(y_pred_test ,y_test)
        print('train rmse: %d'%train_RMSE)
        print('test rmse: %d'%test_RMSE)
        return y_pred_train, y_pred_test


    # TODO: timeseries cross validation (better to work on; not priority for now)
    def cv_timeseries(self, X, y):
        # Approach2: timeseries split
        features = [col for col in all_data_.columns.unique().tolist() if col != 'price_usd']
        target = ['price_usd']

        X = all_data_[features]
        y = all_data_[target]

        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X,y):
            print("TRAIN:", train_index, "TEST:", test_index)    
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        pass


       