from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import os
import sys
import pandas as pd
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))
data_path = str(parent_path) + "/holt_winters/data_processing"

sys.path.append(data_path)


from data_handler import get_data, get_stores, get_families, get_time_series

# single exponential smoothing
def exp_smoothing_forecast(data, forecast_days):
    # create class
    model = SimpleExpSmoothing(data)
    # fit model
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.forecast(forecast_days)
    return yhat

class SimpleExponentialSmoothing(TimeSeriesModel):
    def __init__(self, data, models, parameters):
        super().__init__(data, models, parameters)
        self.parameter_1 = parameters.get("simple_exponential_smoothing_parameter_1", None)
        self.parameter_2 = parameters.get("simple_exponential_smoothing_parameter_2", None)
        # Initialize other attributes specific to simple exponential smoothing model
        def fit_model(self):
        # Implement logic to fit time series model
            pass

        def predict(self):
            # Implement logic to predict using time series model
            pass

class DoubleExponentialSmoothing(TimeSeriesModel):
    def __init__(self, data, models, parameters):
        super().__init__(data, models, parameters)
        self.parameter_1 = parameters.get("double_exponential_smoothing_parameter_1", None)
        self.parameter_2 = parameters.get("double_exponential_smoothing_parameter_2", None)
        # Initialize other attributes specific to double exponential smoothing model
        def fit_model(self):
        # Implement logic to fit time series model
            pass

        def predict(self):
            # Implement logic to predict using time series model
            pass

