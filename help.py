from sklearn.metrics import mean_squared_error
import numpy as np

def sample_data(data, sample_size):
    # e.g.,需要得到50000 data points --> 每330个datapoints中取一个
    interval_range = len(data)//sample_size 
    mid_idx_lst = []
    for i in range(1,sample_size + 1):
        mid_idx = (interval_range*(i-1) + interval_range*i)//2
        mid_idx_lst.append(mid_idx)
    data_sampled = data.iloc[mid_idx_lst]
    return data_sampled

def get_mse(pred, true):
    '''pred -> prediction values, true -> true values '''
    mse = mean_squared_error(true, pred)
    return mse

def get_rmse(pred, true):
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    return rmse

def get_predictions(model, X_train, y_train, X_test, y_test):
    reg = model.fit(X_train, y_train)
    y_pred_train = reg.predict(X_train)
    y_pred_test = reg.predict(X_test)
    train_RMSE = get_rmse(y_pred_train ,y_train)
    test_RMSE = get_rmse(y_pred_test, y_test)
    print('training RMSE:',train_RMSE)
    print('test RMSE:',test_RMSE)
    return y_pred_train, y_pred_test


def extract_time_features(daily_price):
    '''takes in dataframe with "date_time" as index, and "price_usd" as column'''
    daily_price = daily_price.reset_index()
    daily_price['day'] = daily_price['date_time'].apply(lambda x: x.day)
    daily_price['week'] = daily_price['date_time'].apply(lambda x: x.week)
    daily_price['month'] = daily_price['date_time'].apply(lambda x: x.month)
    daily_price['quarter'] = daily_price['date_time'].apply(lambda x: x.quarter)
    return daily_price





