#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from scipy.stats import skew
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency

import seaborn as sns
import matplotlib.pyplot as plt

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


# In[11]:


def draw_missing_data_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data


# In[19]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation score")

    plt.legend(loc="best")
    return plt


# In[20]:


# Plot validation curve
def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name, param_range, cv)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='r', marker='o', markersize=5, label='Training score')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='r')
    plt.plot(param_range, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='Validation score')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='g')
    plt.grid() 
    plt.xscale('log')
    plt.legend(loc='best') 
    plt.xlabel('Parameter') 
    plt.ylabel('Score') 
    plt.ylim(ylim)


# In[99]:


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


# In[21]:


def split_data():
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
    self.test_data = self.data.iloc[(training_size + validation_size): (training_size + validation_size + test_size)]
    return self.train_data, self.val_data, self.test_data


# In[36]:


train = pd.read_csv('./expediadata/train.csv')
test = pd.read_csv('./expediadata/test.csv')
cols_train_only = [col for col in train.columns.unique().tolist() if col not in test.columns.unique().tolist()]
train = train.drop(columns = cols_train_only)
all_data = pd.concat([train, test], ignore_index=True)


# In[37]:


comp_features = ['comp1_rate','comp1_inv','comp1_rate_percent_diff','comp2_inv','comp2_rate','comp2_rate_percent_diff','comp3_rate','comp3_inv','comp3_rate_percent_diff','comp4_rate','comp4_inv','comp4_rate_percent_diff','comp5_rate','comp5_inv','comp5_rate_percent_diff','comp6_rate','comp6_inv','comp6_rate_percent_diff','comp7_rate','comp7_inv','comp7_rate_percent_diff','comp8_rate','comp8_inv','comp8_rate_percent_diff']


# In[38]:


user_features = ['visitor_hist_starrating','visitor_hist_adr_usd','srch_query_affinity_score','orig_destination_distance','site_id','visitor_location_country_id','srch_id']
other_features = ['random_bool']
time_features = ['date_time']


# In[39]:


all_data.columns


# In[40]:


all_data = all_data.drop(columns = comp_features)


# In[41]:


all_data = all_data.drop(columns = user_features)


# In[42]:


all_data = all_data.drop(columns = other_features)


# In[43]:


all_data.columns


# In[44]:


all_data['date_time'] = pd.to_datetime(all_data.date_time)
all_data.sort_values(by=['date_time'],inplace=True)
all_data = all_data.reset_index(drop=True)


# In[46]:


all_data.head()


# In[50]:


country_counts = all_data['prop_country_id'].value_counts()


# In[61]:


d = {}

for ID in country_counts.index:
    d[ID] = country_counts[ID]


# In[63]:


country_id = all_data['prop_country_id'].tolist()


# In[66]:


country_pop = []
for ID in country_id:
    country_pop.append(d[ID])


# In[68]:


all_data['country_value_counts'] = country_pop


# In[70]:


city_counts = all_data['srch_destination_id'].value_counts()


# In[71]:


city = {}
for ID in city_counts.index:
    city[ID] = city_counts[ID]
city_id = all_data['srch_destination_id'].tolist()


# In[72]:


city_pop = []
for ID in city_id:
    city_pop.append(city[ID])


# In[73]:


all_data['city_value_counts'] = city_pop


# In[76]:


all_data_new = all_data.drop(columns = 'prop_country_id')
all_data_new = all_data_new.drop(columns = 'srch_destination_id')


# In[77]:


all_data_new.head()


# In[83]:


prop_116942 = all_data_new[all_data_new['prop_id']==116942]


# In[84]:


prop_116942.head()


# In[85]:


all_data_t = prop_116942.set_index('date_time')
prop_116942_day = all_data_t.resample('D').median()


# In[87]:


prop_116942_day.count()


# In[88]:


draw_missing_data_table(prop_116942_day)


# In[94]:


def split_data(data):
    training_size_large = int(len(data) * 0.8)
    validation_size = int(training_size_large * 0.2)
    training_size = training_size_large - validation_size
    test_size = int(len(data) * 0.2)
    print('training size: %d'%training_size)
    print('validation size: %d'%validation_size)
    print('test size: %d'%test_size)
    # split data by temporal order
    train_data = data.iloc[0: training_size]
    val_data = data.iloc[training_size:(training_size + validation_size)]
    # self.test_data = self.data.iloc[(training_size + validation_size): (training_size + validation_size + test_size)]
    test_data = data.iloc[(training_size + validation_size):]
    return train_data, val_data, test_data


# In[92]:


prop_116942_day = prop_116942_day.drop(columns = 'prop_id')
prop_116942_day.describe()


# In[113]:


X = prop_116942_day[prop_116942_day.loc[:,prop_116942_day.columns != 'price_usd'].columns]
y = prop_116942_day['price_usd']
X_train_data, X_val_data, X_test_data = split_data(X)
y_train_data, y_val_data, y_test_data = split_data(y)


# In[96]:


lr = LinearRegression()
lr.fit(X_train_data, y_train_data)


# In[97]:


scores = cross_val_score(lr, X_train_data, y_train_data, cv=10)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# In[98]:


title = "Learning Curves (LinearRegression)"
cv = 10
plot_learning_curve(lr, title, X_train_data, y_train_data, ylim = (0.7, 1.01), cv = cv, n_jobs = 1)


# In[100]:


variables = [col for col in prop_116942_day.columns.unique().tolist() if col not in ['price_usd']]
model_prop_lr = modeling_pipeline(prop_116942_day, LinearRegression(), variables)
train_data, val_data, test_data = model_prop_lr.split_data()
X_train,y_train,X_val,y_val,X_test,y_test = model_prop_lr.get_X_y()
model_prop_lr.get_modeling_result()


# In[101]:


variables = [col for col in prop_116942_day.columns.unique().tolist() if col not in ['price_usd']]
model_prop_Rg = modeling_pipeline(prop_116942_day, Ridge(), variables)
train_data, val_data, test_data = model_prop_Rg.split_data()
X_train,y_train,X_val,y_val,X_test,y_test = model_prop_Rg.get_X_y()
model_prop_Rg.get_modeling_result()


# In[104]:


variables = [col for col in prop_116942_day.columns.unique().tolist() if col not in ['price_usd']]
model_prop_d = modeling_pipeline(prop_116942_day, tree.DecisionTreeRegressor(), variables)
train_data, val_data, test_data = model_prop_d.split_data()
X_train,y_train,X_val,y_val,X_test,y_test = model_prop_d.get_X_y()
model_prop_d.get_modeling_result()


# In[105]:


X = prop_116942_day[prop_116942_day.loc[:, prop_116942_day.columns != 'price_usd'].columns]  #independent columns
y = prop_116942_day['price_usd']    #target column i.e price range

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns

#get correlations of each features in dataset
corrmat = prop_116942_day.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(prop_116942_day[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[112]:


variables = [col for col in prop_116942_day.columns.unique().tolist() if col not in ['price_usd','prop_location_score1','prop_location_score2','prop_promotion_flag','srch_adults_count','srch_children_count','srch_room_counts','prop_review_score','prop_brand_bool']]
model_prop_d = modeling_pipeline(prop_116942_day, tree.DecisionTreeRegressor(), variables)
train_data, val_data, test_data = model_prop_d.split_data()
X_train,y_train,X_val,y_val,X_test,y_test = model_prop_d.get_X_y()
model_prop_d.get_modeling_result()


# In[ ]:


import xgboost as xgb
variables = [col for col in prop_116942_day.columns.unique().tolist() if col not in ['price_usd','prop_location_score1','prop_location_score2','prop_promotion_flag','srch_adults_count','srch_children_count','srch_room_counts','prop_review_score','prop_brand_bool']]
model_prop_xgb = modeling_pipeline(prop_116942_day, xgb.XGBRegressor(), variables)
train_data, val_data, test_data = model_prop_xgb.split_data()
X_train,y_train,X_val,y_val,X_test,y_test = model_prop_xgb.get_X_y()
model_prop_xgb.get_modeling_result()


# In[ ]:


# # from keras.models import Sequential
# # from keras.layers import Dense, Dropout, LSTM

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LSTM

# # create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(units=16, return_sequences=True, input_shape=(X_train_data.shape[1],1)))
# model.add(LSTM(units=16))
# model.add(Dense(1))

# model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae'])
# history = model.fit(X_train_data, y_train_data, epochs=1,validation_split = 0.25, batch_size=16, verbose=2)


# In[23]:


all_nontime.describe()


# In[25]:


all_nontime['price_usd'].nunique()


# In[27]:


print(all_nontime['price_usd'].unique())


# In[29]:


### find the outliers
all_nontime[all_nontime['price_usd']== all_nontime['price_usd'].max()]


# In[33]:


all_nont = all_nontime.drop(index = 15410513,inplace = False)


# In[34]:


all_nont.describe()


# In[35]:


all_nont[all_nont['price_usd']== all_nont['price_usd'].max()]


# In[36]:


all_nont2 = all_nont.drop(index = 15187983)


# In[37]:


all_nont2.describe()


# In[ ]:


#import numpy as np
#pd.set_option('display.width',1000,'display.max_rows',100000)
all_nont2['price_usd'].value_counts()


# In[50]:


all_count_country = all_nontime.groupby(['prop_country_id']).count()


# In[ ]:


sorted(all_count_country['price_usd'], reverse = True)


# In[56]:


all_nontime.hist(column = 'prop_country_id')


# In[57]:


all_nontime.head()


# In[65]:


country_225 = all_nontime[all_nontime['prop_country_id']==225]


# In[66]:


country_225.describe()


# In[61]:


draw_missing_data_table(country_225)


# In[69]:


# fill na to medium
country_225 = country_225.fillna(country_225.median())


# In[70]:


draw_missing_data_table(country_225)


# In[94]:


variables = [col for col in country_225.columns.unique().tolist() if col not in ['price_usd','prop_country_id']]
model_prop_lr = modeling_pipeline(country_225, LinearRegression(), variables)
train_data_225, val_data_225, test_data_225 = model_prop_lr.split_data()
X_train_225,y_train_225,X_val_225,y_val_225,X_test_225,y_test_225 = model_prop_lr.get_X_y()
model_prop_lr.get_modeling_result()


# In[95]:


#variables = [col for col in country_225 if col not in ['price_usd','prop_country_id']]
model_prop_ridge = modeling_pipeline(country_225, Ridge(), variables)
train_data_225, val_data_225, test_data_225 = model_prop_ridge.split_data()
X_train_225,y_train_225,X_val_225,y_val_225,X_test_225,y_test_225 = model_prop_ridge.get_X_y()
model_prop_ridge.get_modeling_result()


# In[96]:



from sklearn import tree
#variables = [col for col in country_225 if col not in ['price_usd','prop_country_id']]
model_prop_tree = modeling_pipeline(country_225, tree.DecisionTreeRegressor(), variables)
train_data_225, val_data_225, test_data_225 = model_prop_tree.split_data()
X_train_225,y_train_225,X_val_225,y_val_225,X_test_225,y_test_225 = model_prop_tree.get_X_y()
model_prop_tree.get_modeling_result()


# In[101]:



X = country_225[country_225.loc[:, country_225.columns != 'price_usd'].columns]  #independent columns
y = country_225['price_usd']    #target column i.e price range
# from sklearn.ensemble import ExtraTreesClassifier
# import matplotlib.pyplot as plt
# model = ExtraTreesClassifier()
# model.fit(X,y)
# print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
# #plot graph of feature importances for better visualization
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(10).plot(kind='barh')
# plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns

#apply SelectKBest class to extract top 10 best features
# bestfeatures = SelectKBest(score_func=chi2, k=10)
# fit = bestfeatures.fit(X,y)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X.columns)
# #concat two dataframes for better visualization 
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']  #naming the dataframe columns
# print(featureScores.nlargest(10,'Score'))  #print 10 best features

#get correlations of each features in dataset
corrmat = country_225.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(country_225[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[71]:


X = country_225[country_225.loc[:, country_225.columns != 'price_usd'].columns]
y = country_225['price_usd']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)


# In[72]:


# Debug
print('Inputs: \n', X_train.head())
print('Outputs: \n', y_train.head())


# In[76]:


# Fit logistic regression
linreg = LinearRegression()
linreg.fit(X_train, y_train)


# In[78]:


scores = cross_val_score(linreg, X_train, y_train, cv=10)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# In[102]:


#### set the price range from 10 dollar to 100,000  df.loc[(df['column_name'] >= A) & (df['column_name'] <= B)]
data_price = country_225.loc[(country_225['price_usd']>=10)&(country_225['price_usd']<=100000)]


# In[103]:


data_price.describe()


# In[104]:



model_price_lr = modeling_pipeline(data_price, LinearRegression(), variables)
train_data_225, val_data_225, test_data_225 = model_price_lr.split_data()
X_train_225,y_train_225,X_val_225,y_val_225,X_test_225,y_test_225 = model_price_lr.get_X_y()
model_price_lr.get_modeling_result()


# In[105]:


variables = ['prop_id','prop_starrating','prop_review_score','prop_log_historical_price','srch_destination_id','srch_room_count','srch_saturday_night_bool']
model_price_lr = modeling_pipeline(data_price, LinearRegression(), variables)
train_data_225, val_data_225, test_data_225 = model_price_lr.split_data()
X_train_225,y_train_225,X_val_225,y_val_225,X_test_225,y_test_225 = model_price_lr.get_X_y()
model_price_lr.get_modeling_result()


# In[ ]:


### select features: prop_id, prop_starrating, prop_review_score, prop_brand_bool, prop_log_historical_price

