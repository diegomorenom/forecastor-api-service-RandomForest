from forecast_generator import forecast_process as fp

import json
import os
import sys
import itertools
from tqdm import tqdm
from time import sleep

import warnings
warnings.filterwarnings('ignore')
#import pprint

path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))
data_path = str(parent_path) + "/forecastor/data_processing"
forecast_path = str(parent_path) + "/forecastor/forecast_generator"

sys.path.append(data_path)
sys.path.append(forecast_path)


from data_handler import get_data

# Load JSON file
with open("parameters.json", "r") as file:
    info = json.load(file)

# Prepare your data, models, parameters, and forecast_days
data = get_data()
models = ['HoltWinters']#info[0]["Models"]
forecast_days = info[0]["ForecastDays"]
parameters = info[1]

# Instantiate a class that inherits from BaseForecastingProcess
forecast_process_instance = fp.ForecastingProcess(data, models, parameters, forecast_days)

# Call the run_all_models method
forecast_process_instance.run_all_models()