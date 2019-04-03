$Project Goal
=============

This is a project for mastery course SI699. The data that we would be using is available on https://www.kaggle.com/c/expedia-personalized-sort/data.


$Codes Explanation
==================

    [About basic modeling pipeline]:
    -------------------------------
    
        basic_model_pipeline_v2.ipynb:
            - use 5000 examples
            - basic code pipeline for modeling, try basic regression model, e.g., linear regression, ridge regression, etc.

        basic_model_pipeline_v3.ipynb:
            - using 10000 examples 
            - basic code pipeline for modeling, try basic regression model, e.g., linear regression, ridge regression, etc.
        
        
        
    [seq2seq folder]:
    ----------------
    
        storing codes that are relevant to seq2seq implementation
        - seq2seq_v3.ipynb:
            - Currently working on resampling data by days, input would be the previous seven days' hotel price, and the output would be hotel prices of the next seven days

        - seq2seq_tutorial.ipynb:
            - reference from online tutorial about using seq2seq to predict stock market


    [About time features modeling]:
    ------------------------------
         
        time_feature_modeling.py:
            - aggregate price data by each property id
            - extract time features (day,month,quarter,etc.) and observe the price variations by each time features; ideally we would want to see enough variations by each of these time features, so that we can be confident to build regression model upon them
            - plot price data trend --> the goal is to observe day signal
            - build regression model based on the time features that we extracted
        
        ts_modeling_v2.py:
            - define main functionality to implement simple timeseries prediction using AR, MA, and ARIMA
            - used for ts_modeling_v2.ipynb


        time_feature_modeling.ipynb:
            - main program to implement codes in time_feature_modeling.py
        
        timeseries_modeling_v1.ipynb:
            - main program tp implement 'moving_average.py'
            - reference from si671 project codes (more to see in github)
            - TODO: this only get the training error; need to have a better understanding about the code logics behind and get out-sample error
        
        ts_modeling_v2.ipynb: (MOST RECENT)
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
        
        
        

    [Other codes]:
    -------------
    
        modeling-trial2.ipynb:
            - feature selection: categorical variables check p-value and chi2 score
      
        
$TODO
=====
    - Learn more about emsembel methods <-- stacking models       
    

