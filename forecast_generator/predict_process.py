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
data_path = str(parent_path) + "/neural_network/data_processing"
model_path = str(parent_path) + "/neural_network/modeling"

sys.path.append(data_path)
sys.path.append(model_path)


from data_handler import get_data, get_stores, get_families, get_time_series, get_splitted_df, fill_values, structure_predictions, save_predictions
from data_modeling import model_data
from neural_network import neural_network_forecast

forecast_days = 30


def run_forecast():
        print("Getting and transforming data")
        df = get_data()
        stores = [1]#get_stores(df)
        families = ['GROCERY I']#, 'BEVERAGES', 'PRODUCE', 'CLEANING','DAIRY'] # get_families(df)

        print("Iterating Forecast")
        #iterations = len(itertools.product(stores, families))
        for i in tqdm(range(len(stores))):
                for s in stores:
                        for f in families:
                                df_info = get_splitted_df(df, f, s)
                                df_ts = get_time_series(df_info)
                                df_ts = fill_values(df_ts)
                                df_reg, scaler = model_data(df_ts)
                                if not df_reg.empty and df_ts['sales'].sum() > 0:
                                        df_yhat = neural_network_forecast(df_reg, forecast_days, scaler)
                                        df_pred = structure_predictions(df['date'].max(), df_yhat, f, s)
                                        save_predictions(df['date'].max(), df_pred)
                                else:
                                        pass
                sleep(0.02)
        
        return print("Forecast finished")




        





