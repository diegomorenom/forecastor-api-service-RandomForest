from statsmodels.tsa.holtwinters import ExponentialSmoothing

import os
import sys
import pandas as pd

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))
forecast_path = str(parent_path) + "/forecastor-api-service/forecast_generator"
sys.path.append(forecast_path)

    
class HoltWinters:
    def __init__(self, data, parameters, forecast_days):
        self.data = data 
        self.forecast_days = forecast_days 
        self.parameters = parameters
        self.seasonal = self.parameters['model_parameters']['seasonal']
        self.seasonal_periods = self.parameters['model_parameters']['seasonal_periods']
        
    def fit_model(self):
        model = ExponentialSmoothing(self.data, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods)
        fitted_model = model.fit()
        return fitted_model

    def predict(self, fitted_model):
        yhat = fitted_model.forecast(self.forecast_days)
        return yhat