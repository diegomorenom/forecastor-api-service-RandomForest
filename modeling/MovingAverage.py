from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta

import os
import sys
import pandas as pd

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))


# simple moving average
def moving_average_model(data, forecast_days):
    y_hat_sma = data.copy()
    y_hat_sma['forecast'] = data['prediction_column'].rolling(forecast_days).mean()
    return y_hat_sma

class MovingAverage:
    def __init__(self, data, parameters, forecast_days):
        self.data = data 
        self.forecast_days = forecast_days 
        self.parameters = parameters
        self.time_window = self.parameters['model_parameters']['time_window']
        
    def fit_model(self):
        fitted_model = "No need to fit for moving average"
        return fitted_model

    def predict(self, fitted_model):
        print(fitted_model)
        data_df = self.data.copy()
        for d in range(self.forecast_days):
            df_yhat =  moving_average_model(data_df, self.forecast_days)
            last_value = df_yhat['forecast'].iloc[-1]
            df_forecast = df_yhat.reset_index()
            forecast_date = df_forecast.date.max() + timedelta(days=1)
            df_forecast = df_forecast[['date','prediction_column']]
            df_forecast = df_forecast._append(pd.DataFrame({"date":[forecast_date],"prediction_column":[last_value]}),ignore_index=True)
            df_forecast = df_forecast.set_index('date')
            data_df = df_forecast.copy()

        return df_forecast

