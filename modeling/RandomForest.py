
from sklearn.ensemble import RandomForestRegressor

import os
import sys
import pandas as pd
import numpy as np
from datetime import timedelta


path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))
data_path = str(parent_path) + "/forecastor-api-service/data_processing"
sys.path.append(data_path)
from data_modeling import scale_back_data, labels_pred

class RandomForest():
    def __init__(self, data, scaler, parameters, forecast_days):
        self.data = data 
        self.scaler = scaler
        self.forecast_days = forecast_days
        self.parameters = parameters
        self.n_estimators = self.parameters['model_parameters']['n_estimators']
        self.random_state = self.parameters['model_parameters']['random_state']
    
    def fit_model(self):
        # Labels are the values we want to predict
        labels = np.array(self.data['var1(t)'])
        # Remove the labels from the features
        # axis 1 refers to the columns
        features= self.data.drop('var1(t)', axis = 1)
        # Saving feature names for later use
        feature_list = list(features.columns)
        # Convert to numpy array
        features = np.array(features)
        # Instantiate model with 1000 decision trees
        rf = RandomForestRegressor(n_estimators = self.n_estimators, random_state = self.random_state)
        # Train the model on training data
        rf.fit(features, labels)
        return rf

    def predict(self, random_forest_model):
        df_forecast = self.data.copy()
        columns = len(self.data.columns)
        last_row = list(self.data.values[-1].tolist())
        print('Making predictions')
        for d in range(self.forecast_days):
            features = labels_pred(last_row)
            features = np.array(features).reshape((1, columns-1))
            prediction = random_forest_model.predict(features)
            del last_row[0] 
            last_row.append(prediction[0])
            forecast_date = df_forecast.index.max() + timedelta(days=1)
            df_forecast.loc[forecast_date] = last_row 
        df_pred = scale_back_data(df_forecast, self.scaler)
        df_pred = df_pred[df_pred.columns[-1]]
        return df_pred
