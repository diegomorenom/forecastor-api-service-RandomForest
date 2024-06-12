import prophet

import os
import sys
import pandas as pd

path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))
forecast_path = str(parent_path) + "/forecastor-api-service/forecast_generator"
sys.path.append(forecast_path)

    
class FacebookProphet:
    def __init__(self, data, parameters, forecast_days):
        self.data = data
        self.forecast_days = forecast_days 
        self.parameters = parameters
        self.interval_width = self.parameters['model_parameters']['interval_width']
        self.pred_freq = self.parameters['model_parameters']['pred_freq']
        
    def fit_model(self):
        df = self.data.copy()
        df = df.reset_index()
        df.rename(columns={'date': 'ds', 'prediction_column': 'y'}, inplace = True)
        df['ds'] = df['ds'].dt.to_timestamp('s').dt.strftime('%Y-%m-%d %H:%M:%S.000')
        model = prophet.Prophet(interval_width=0.95)
        fitted_model = model.fit(df)
        return fitted_model

    def predict(self, fitted_model):
        future_dates = fitted_model.make_future_dataframe(periods=self.forecast_days, freq=self.pred_freq)
        yhat = fitted_model.predict(future_dates)
        yhat = yhat[['ds', 'yhat']].set_index('ds')
        return yhat