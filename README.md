# Project Goal

This is a project for mastery course SI699. The data that we would be using is available on https://www.kaggle.com/c/expedia-personalized-sort/data.


# Codes Explanation
### About modeling pipeline in general: 
* basic_model_pipeline_v2.ipynb:
    - use 5000 examples
    - basic code pipeline for modeling, try basic regression model, e.g., linear regression, ridge regression, etc.

* basic_model_pipeline_v3.ipynb:
    - using 10000 examples 
    - basic code pipeline for modeling, try basic regression model, e.g., linear regression, ridge regression, etc.

        
* sample_prop.ipynb: _(updated: 04/13/2019)_
    - this file is used to get a sample pool of properties; currently randomly sample 1000 prop id from the entire dataset, and get the distribution of daily data records of each prop id. Then set 75% percentile for choosing the higher bound of picking prop id, we would say that only prop id with valid records larger than this threshold would be considered for prediction.
        
        
### seq2seq folder
storing codes that are relevant to seq2seq implementation
* seq2seq_v3.ipynb:
    - Currently working on resampling data by days, input would be the previous seven days' hotel price, and the output would be hotel prices of the next seven days

* seq2seq_tutorial.ipynb:
    - reference from online tutorial about using seq2seq to predict stock market


### About time features modeling

* time_feature_modeling.py:
    - aggregate price data by each property id
    - extract time features (day,month,quarter,etc.) and observe the price variations by each time features; ideally we would want to see enough variations by each of these time features, so that we can be confident to build regression model upon them
    - plot price data trend --> the goal is to observe day signal
    - build regression model based on the time features that we extracted
        
* ts_modeling_v2.py:
    - define main functionality to implement simple timeseries prediction using AR, MA, and ARIMA
    - used for ts_modeling_v2.ipynb

* time_feature_modeling.ipynb:
    - main program to implement codes in time_feature_modeling.py

* time_feature_modeling_v2.ipynb:
    - updated: organize codes and add "week" feature into regression model
        
* timeseries_modeling_v1.ipynb:
    - main program tp implement 'moving_average.py'
    - reference from si671 project codes (more to see in github)
    - TODO: this only get the training error; need to have a better understanding about the code logics behind and get out-sample error
        
* ts_modeling_v2.ipynb: (by destination + by starratings)
    - using AR, MA, and ARIMA model + implement Grid search for each model: ARIMA model performs best; parameter sets p,d,q =(2,1,1)
    - stacking ARIMA model and regression modeling (trying Linear regression, ridge, XGboost for second-layer regression)
    - in use with "ts_modeling_v2.py"
    - TODO:
            - in-sample error vs. out-sample error
            - log-transform price data
            - timeseries cross validation
            - regression model split data offset by 1 (handle the mismatch dimension when trying to concatenate results from two different models)
    - reference: 
            - https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
            - https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/
            - stats model: https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima_model.ARIMA.html

* ts_modeling_v3.ipynb: (by property)
        - updated: fix the error of unmatch dimensions of ARIMA model and regression model
        - updated: ts_modeling_v2.ipynb - predict price by destination + by starrating
                        vs. ts_modeling_v3.ipynb - predict price by property_id
        - updated: rewrite functions into a class, that allows you to enter a prop_id, and get the predicted pricestacking
        
* time_property_modeling.ipynb: 
        - combine time modeling + property feature modeling
        - TODO: seem to overfit => can try stack three models directly, instead of fit a second-layer model for time and combine with property feature modeling
        
        
        
        
* time_cv.ipynb: _(updated: 04/13/2019)_
        - TODO: apply timeseries cross validation for ARIMA modeling
        - problem: 
        train(e.g., 65 examples) => predict test (e.g., 64 examples) 
        => ARIMA fail to predict becasue of insufficient degree of freedom
        vs. train (0-10) -> test (11)
            train (0-11) -> test(12)

### About property feature modeling
* prop_modeling.py:
        - modified based on features_prop_update.ipynb 
        - rewrite functions into class

    
### About property + time feature modeling
* time_property_modeling_v2.ipynb
        - TODO: predict for each property
            
    

### Other codes
* modeling-trial2.ipynb:
        - feature selection: categorical variables check p-value and chi2 score
            
      
# Data files
#### experdia_data folder:
    - train.csv
    - test.csv
    - all_data.csv (not remember how to get this in the first place; however, the total length of records not match train and test combined)

#### res folder:
    - ElasticNet_y_pred_XX.csv => from Crtystal property feature modeling prediction results
    - sampled_data_1000000.csv => sampled data to build code pipelines

    

