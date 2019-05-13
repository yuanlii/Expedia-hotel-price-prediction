# Project Goal

The overall goal of this project is to build a hotel room rate prediction system that helps customers to evaluate the price and determine the best time to book a room for traveling. Several questions that we would like to answer include:
* When we book hotel, how can we know the price is reasonable or overcharged? 
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
    
    
#### 5. Log transform skewed data

First, we compute the skewness for each numeric variable. We defined variables with skewness > 0.75 as "highly skewed", and we would log transformed those variables with high skewness to make them more normaly distributed. 
   * more about skewness: For normally distributed data, the skewness should be about 0. For unimodal continuous distributions, a skewness value > 0 means that there is more weight in the right tail of the distribution. ([reference]((https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html)))

![skewed price data](https://github.com/yuanlii/Expedia_hotel_price_prediction/blob/master/pictures/skewed_price_log_transformed.png)
    
#### 2. Outlier value detection

Hotel room rates can have as low as $0.2/night, and as high as $+5m/night; we remove those outliers that are significantly deviated from the rest of hotel room rate distribution
    
    
#### 3. Convert categorical variables to continuous variables

For categorical variables with more than 100 instances, e.g., country_id, destination_id, property_id, etc. it wouldn't make sense to one-hot encoding them all; so what we did is to compute the popularity, i.e., how many times each instance of a category ever appear in the dataset to represent the instance itself. For example, for property_id = 116942, we count how many records with property_id = 116942 are there in the dataset, and use that continuous number to represent property_id = 116942. Same logic and transformation is applied to country_id and destination_id as well as other categorical variables.


#### 4. Aggregate data based on time range

Our ultimate goal is to predict hotel room rate for one property listing in one single day. However, from the Expedia dataset, it only lists the data per user search and potential at multiple timestamps within a day, so we would need to aggregate the data by day. 

![daily price trend](https://github.com/yuanlii/Expedia_hotel_price_prediction/blob/master/pictures/daily_price_trend.png)



## Feature Importance

In order to understand the importance of each feature, we use XGBoost to get the importance of each feature:

![XGBoost feature importance](https://github.com/yuanlii/Expedia_hotel_price_prediction/blob/master/pictures/feature_importance_XGBosot.png)

From which we can tell that prop_country_id, prop_log_historical_price and prop_review_score are the top 3 most importance features. This diagram gives us an understanding of what are the important features in terms of building model for the next stage.


## Modeling

TODO: more about rationale to adopt multi-layer modeling

We applied a multi-layer modeling approach to resolve the complexity of the problem, by dividing them into several subproblems that are easier to tackle. First, we try to divide features by its nature into several feature groups, including User, Property, Time, and Competitors. We would then build model for each of the feature group (which refered as "first layer modeling"). After modeling selection for each feature group, including hyperparameter tuning and cross validation, we are able to get the best predictions based on each feature group. Then we would concatenate the predictions from each feature group modeling, and use as input to fit a second-layer model. 

![modeling structure](https://github.com/yuanlii/Expedia_hotel_price_prediction/blob/master/pictures/modeling_structure.png)

## Results
### 1. Property Modeling

For property feature group modeling, we fit several linear and non-linear models inclusing Ridge, Random Forest, Elastic Net, etc., and compare the performance of each of the model. Elastic Net turns out to have the best performance in modeling property features.
   * TODO: more about Elastic Net

![property modeling results](https://github.com/yuanlii/Expedia_hotel_price_prediction/blob/master/pictures/prop_modeling_result.png)

### 2. Time Modeling

For time feature modeling, we would use ARIMA. 

![Autocorrelation](https://github.com/yuanlii/Expedia_hotel_price_prediction/blob/master/pictures/autocorrelation_v3.png)

![Time-series Modeling](https://github.com/yuanlii/Expedia_hotel_price_prediction/blob/master/pictures/ts_prediction.png)

![time prediction results](https://github.com/yuanlii/Expedia_hotel_price_prediction/blob/master/pictures/time_modeling_v2.png)

### 3. Multi-layer Modeling

## Conclusion


## More Exploration


## Reflections




    
    
