readme: 

# have built basic modeling pipeline, codes include:
     # model_pipeline_v1.ipynb:
        - build basic modeling pipeline, include load data, data preprocessing, modeling and model evaluation
        - baseline method include linear regression, ridge regression
     
     # model_pipeline_v2.ipynb:
         - add XGBoost implementation (for implementation GBDT)
         - add ridge regression (for continuous variable only and for normalized continuous variables)
     
# codes that can be deprecated or archived:
    - GBDT.ipynb: updated implementation of GBDT has been added to model_pipeline_v2.ipynb
    

# explanation about Source and Output files:
    - res/sampled_data_5000.csv:
        - result of sampled data that contain 5000 examples from the entire dataset; 
        - not yet clean, and just raw data; 
        - mainly used to build code pipeline
        
    - res/price_data_8192.csv:
        - choose destination id = 8192, get the daily price of this particular destination as example to build seq2seq pipeline

# Variable explanation:
    - 'price_usd': is the listed price of searching
    - 'srch_destination_id': is the city that user search
        - primary goal: is to provide reference to users who search similar destinations
     
     
    

    
  
