from xgboost import XGBRegressor

import os
import sys
import pandas as pd
from pmdarima.model_selection import train_test_split as time_train_test_split

path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))
forecast_path = str(parent_path) + "/forecastor-api-service/forecast_generator"
sys.path.append(forecast_path)
data_path = str(parent_path) + "/forecastor-api-service/data_processing"
sys.path.append(data_path)
database_path = str(parent_path) + "/forecastor-api-service/data_processing/data_base"
sys.path.append(database_path)
from data_modeling import train_test_split


def create_features(df, target_variable):
    """
    Creates time series features from datetime index
    
    Args:
        df (float64): Values to be added to the model incl. corresponding datetime
                      , numpy array of floats
        target_variable (string): Name of the target variable within df   
    
    Returns:
        X (int): Extracted values from datetime index, dataframe
        y (int): Values of target variable, numpy array of integers
    """
    print(df.dtypes)
    #df['date'] = df.index
    #df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if target_variable:
        y = df[target_variable]
        return X, y
    return X

    
class XGBoost:
    def __init__(self, data, parameters, forecast_days):
        self.data = data
        self.forecast_days = forecast_days 
        self.parameters = parameters
        self.interval_width = self.parameters['model_parameters']['interval_width']
        self.pred_freq = self.parameters['model_parameters']['pred_freq']
        
    def fit_model(self):
        df = self.data.copy()
        df = df.reset_index()
        df_train, df_test = time_train_test_split(df, test_size=int(len(df)*0.2))
        trainX, trainY = create_features(df_train, 'prediction_column')
        testX, testY = create_features(df_test, target_variable='prediction_column')
        xgb = XGBRegressor(objective= 'reg:linear', n_estimators=1000)
        xgb.fit(trainX, trainY,
                eval_set=[(trainX, trainY), (testX, testY)],
                early_stopping_rounds=50,
                verbose=True)
        return xgb

    def predict(self, fitted_model):
        df_test = pd.read_csv(database_path+"/data_test.csv")
        df_test.rename({'forecast_date': 'date'}, axis=1, inplace=True)
        df_test["date"] =pd.to_datetime(df_test["date"])
        df_test["date"] = df_test["date"].dt.to_period("D")
        testX, testY = create_features(df_test, 'forecast')
        yhat = fitted_model.predict(testX)
        df_test['yhat'] = yhat
        df_yhat = df_test[['date', 'yhat']].set_index('date')
        print(df_yhat)
        return df_yhat