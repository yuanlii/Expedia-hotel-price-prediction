# Project Goal

The overall goal of this project is to build a hotel room rate prediction system that helps customers to evaluate the price and determine the best time to book a room for traveling. Several questions that we would like to answer include:
* When we book a hotel, how can we know the price is reasonable or overcharged? 
* Can we have a number to benchmark?
* Can we know the fluctuation of hotel room price by season?
These are the questions that we want to answer throughout this project.


## Data Overview

We use Personalize Expedia hotel searches – ICDM 2013 from online Kaggle competition ( > 4GB), which includes a wide variety of data on User, Property, time, competitor, etc. It contained nearly 10 million historical hotel search results representing approximately 400 thousand unique search queries on the popular travel booking website Expedia.com. 
* Features: ~ 54 features, which can be divided into several sub-categories:
    * Search (time, location, etc.) 
    * User (country, historical data, etc.) 
    * Hotel (price, star, reviews, etc.)


## Data Preprocess

#### 1. Handle missing data
* To handle variables with missing data > 50%, we would create a binary variable as indictor of whether the data is missing or not for a specific data record 
* If variable with missing data < 50%, we would replace NA values with median values
    
    
#### 2. Log transform skewed data

First, we compute the skewness for each numeric variable. We defined variables with skewness > 0.75 as "highly skewed", and we would log transformed those variables with high skewness to make them more normaly distributed. 
   * more about skewness: For normally distributed data, the skewness should be about 0. For unimodal continuous distributions, a skewness value > 0 means that there is more weight in the right tail of the distribution. ([reference]((https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html)))

![skewed price data](https://github.com/yuanlii/Expedia_hotel_price_prediction/blob/master/pictures/skewed_price_log_transformed.png)
    
#### 3. Outlier value detection

Hotel room rates can have as low as $0.2/night, and as high as $+5m/night; we remove those outliers that are significantly deviated from the rest of hotel room rate distribution
    
    
#### 4. Convert categorical variables to continuous variables

For categorical variables with more than 100 instances, e.g., country_id, destination_id, property_id, etc. it wouldn't make sense to one-hot encoding them all; so what we did is to compute the popularity, i.e., how many times each instance of a category ever appear in the dataset to represent the instance itself. For example, for property_id = 116942, we count how many records with property_id = 116942 are there in the dataset, and use that continuous number to represent property_id = 116942. Same logic and transformation is applied to country_id and destination_id as well as other categorical variables.


#### 5. Aggregate data based on time range

Our ultimate goal is to predict hotel room rate for one property listing in one single day. However, from the Expedia dataset, it only lists the data per user search and potential at multiple timestamps within a day, so we would need to aggregate the data by day. 

![daily price trend](https://github.com/yuanlii/Expedia_hotel_price_prediction/blob/master/pictures/daily_price_trend.png)


#### 6. Split data by time 

Sort data by time, and split data into training, validation and test set.

![train test split](https://github.com/yuanlii/Expedia_hotel_price_prediction/blob/master/pictures/train_test_split.png)


## Explore Feature Importance

In order to understand the importance of each feature, we use XGBoost to get the importance of each feature:

![XGBoost feature importance](https://github.com/yuanlii/Expedia_hotel_price_prediction/blob/master/pictures/feature_importance_XGBosot.png)

From which we can tell that prop_country_id, prop_log_historical_price and prop_review_score are the top 3 most importance features. This diagram gives us an understanding of what are the important features in terms of building model for the next stage.


## Modeling Methodology

TODO: more about rationale to adopt multi-layer modeling

We applied a multi-layer modeling approach to resolve the complexity of the problem, by dividing them into several subproblems that are easier to tackle. First, we try to divide features by its nature into several feature groups, including User, Property, Time, and Competitors. We would then build model for each of the feature group (which refered as "first layer modeling"). After modeling selection for each feature group, including hyperparameter tuning and cross validation, we are able to get the best predictions based on each feature group. Then we would concatenate the predictions from each feature group modeling, and use as input to fit a second-layer model. 

![modeling structure](https://github.com/yuanlii/Expedia_hotel_price_prediction/blob/master/pictures/modeling_structure.png)

How to implement such modeling pipeline using Python? After we get prediction from each of the model, first we need to pay attention to its format. If it is formatted as an ndarray, we would need to reshape it into array with shape (-1,1), and stack each formatted prediction vertically by column. Illustration below may further explain the entire process. 

![second-layer modeling](https://github.com/yuanlii/Expedia_hotel_price_prediction/blob/master/pictures/multi-layer_modeling.png)

python codes snippet: _(complete code please see: ts_modeling_v2.py)_

```python
regression_y_pred_val = self.regression_y_pred_val.reshape(-1,1)
regression_y_pred_test = self.regression_y_pred_test.reshape(-1,1)

ARIMA_val_predictions = np.array(self.ARIMA_val_predictions).reshape(-1,1)
ARIMA_test_predictions = np.array(self.ARIMA_test_predictions).reshape(-1,1)

X_train = np.concatenate(( regression_y_pred_val, ARIMA_val_predictions), axis=1)
X_test = np.concatenate(( regression_y_pred_test, ARIMA_test_predictions), axis=1)
```

more explanation on ts + prop => validation as train, and test as test;
timeseries cross validation



## More Exploration

* autoencoder
* seq2seq


If you want to know more about this project, please check our poster:

![presentation poster](https://github.com/yuanlii/Expedia-hotel-price prediction/blob/master/pictures/poster_si699_yuan%26Jing.pdf)




    
    
