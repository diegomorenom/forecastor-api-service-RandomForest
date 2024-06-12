from abc import ABC, abstractmethod

import itertools
import os
import sys
from tqdm import tqdm
from time import sleep

import importlib

import warnings
warnings.filterwarnings('ignore')
#import pprint

path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))
data_path = str(parent_path) + "/forecastor-api-service/data_processing"
modeling_path = str(parent_path) + "/forecastor-api-service/modeling"

sys.path.append(data_path)
sys.path.append(modeling_path)


from data_handler import get_time_series, get_splitted_df, fill_values, structure_predictions, save_predictions, get_train_test, delete_forecast_files
from data_modeling import model_data, get_error_metrics


class BaseForecastingProcess(ABC):
    def __init__(self, data, models, parameters, forecast_days, forecasting_type):
        self.data = data
        self.models = models
        self.parameters = parameters
        self.forecast_days = forecast_days
        self.forecasting_type = forecasting_type

    @abstractmethod
    def process_data(self):
        pass

    @abstractmethod
    def run_all_models(self):
        pass

    @abstractmethod
    def save_forecast(self, df_pred):
        pass

class ForecastingProcess(BaseForecastingProcess):
    def __init__(self, data, models, parameters, forecast_days, forecasting_type):
        super().__init__(data, models, parameters, forecast_days, forecasting_type)
        self.data = data
        self.models = models
        self.parameters = parameters
        self.forecast_days = forecast_days
        self.forecasting_type = forecasting_type
        # Filter time series and regression models
        #self.time_series_models = [model for model in models if issubclass(model, TimeSeriesModel)]
        #self.regression_models = [model for model in models if issubclass(model, RegressionModel)]
    
    def process_data(self):
        df_info = get_splitted_df(self.data)
        df_ts = get_time_series(df_info)
        df_ts = fill_values(df_ts)
        train = get_train_test(df_ts, self.forecasting_type)
        return train

    def run_all_models(self):
        df_ts = self.process_data()
        for model in self.models:
            print("Predicting "+model)
            if model == self.models[0]:
                delete_forecast_files()
            model_parameters = self.parameters[model]
            module = importlib.import_module(model)
            model_class = getattr(module, model)
            model_type = model_parameters['model_type']
            if model_type == 'TimeSeries':
                model_instance = model_class(df_ts, model_parameters, self.forecast_days)
            elif model_type == 'Regression':
                df_reg, scaler = model_data(df_ts)
                model_instance = model_class(df_reg, scaler, model_parameters, self.forecast_days)
            
            fitted_model = model_instance.fit_model()
            df_yhat = model_instance.predict(fitted_model)
            
            self.save_forecast(df_ts, df_yhat, model_parameters['model_name'])
        if self.forecasting_type == "historicalForecasting":
            errors = get_error_metrics(self.models)
            print(errors)

    def save_forecast(self, df_ts, df_pred, model):
        max_date = str(df_ts.reset_index()['date'].max())
        df_pred = structure_predictions(max_date, df_pred, model)
        save_predictions(max_date, df_pred, model)

class TimeSeriesModel(ForecastingProcess):
    def __init__(self, data, models, parameters, forecast_days):
        super().__init__(data, models, parameters, forecast_days)

    def prepare_data(self):
        # Implement logic to prepare time series data
        pass

    

class RegressionModel(ForecastingProcess):
    def __init__(self, data, models, parameters, forecast_days):
        super().__init__(data, models, parameters, forecast_days)

    def prepare_data_reg(self):
        # Implement logic to prepare data for regression model
        pass
    
