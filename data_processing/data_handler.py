import pandas as pd

import os
import datetime 
import json
import shutil

path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))
data_base_path = str(parent_path)+"/forecastor-api-service/data_processing/data_base"
forecast_path = str(parent_path)+"/forecastor-api-service/data_processing/forecast_files"
forecast_info_file = str(parent_path) + "/forecastor-api-service/forecast_info.JSON"



def get_data():
    data = pd.read_csv(data_base_path+"/data_api.csv")
    data = data[['date_column', 'prediction_column']]
    data.columns = ['date', 'prediction_column']
    return data

def get_splitted_df(data):
    #df_info = data[(data['family']==family)&(data['store_nbr']==store_nbr)]
    df_info = pd.pivot_table(data, values='prediction_column', index=['date'], aggfunc="sum").reset_index()
    df_info = df_info[['date', 'prediction_column']]
    return df_info

def get_time_series(df_info):
    df_info['date'] = pd.to_datetime(df_info['date'])
    date_range = pd.date_range(df_info['date'].min(),df_info['date'].max(),freq='d')
    df_ts = pd.DataFrame({'date':date_range})
    df_ts = df_ts.merge(df_info, how='left', on='date')
    df_ts = df_ts.set_index('date')
    df_ts.index = pd.DatetimeIndex(df_ts.index).to_period('D')
    return df_ts

def fill_values(df_ts):
    df_ts = df_ts.fillna(0)
    return df_ts

def get_stores(data):
    stores = list(data['store_nbr'].unique())
    return stores

def get_families(data):
    families = list(data['family'].unique())
    return families

def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

def structure_predictions(date, df_pred, model):
    df_pred = pd.DataFrame(df_pred).reset_index()#, columns=['forecast_date','forecast'])
    df_pred.columns = ['forecast_date','forecast']
    df_pred['date'] = date
    df_pred['model'] = model
    #df_pred['family'] = family
    #df_pred['store_nbr'] = store_nbr
    df_pred['date_updated'] = datetime.datetime.now()
    return df_pred
    
def save_predictions(date, df_pred, model):
    #file_name = forecast_path+'/forecast_'+str(model)+'_'+str(date.replace('-', ''))+'.csv'
    file_name = forecast_path+'/forecast.csv'
    df_pred['forecast_date'] = df_pred['forecast_date'].astype(str)
    df_pred['forecast_date'] = pd.to_datetime(df_pred['forecast_date'])
    df_pred = df_pred.loc[df_pred['forecast_date'] > date]
    if os.path.isfile(file_name):
        df_pred.to_csv(file_name, mode='a', index=False, header=False)
    else:
        df_pred.to_csv(file_name, mode='w', index=False, header=True)

    return "Forecast saved"

def get_train_test(data, forecasting_type):
    with open(forecast_info_file) as f:
        forecast_info = json.load(f)

    if forecasting_type == "historicalForecasting": 
        forecast_days = forecast_info['forecastDays']

        model_data_long = data.shape
        model_data_long = model_data_long[0]

        n_train_days = round(model_data_long-forecast_days )
        train = data.iloc[:n_train_days, :]
        test = data.iloc[n_train_days:, :]

        test = test.reset_index()
        test.columns = [['forecast_date', 'forecast']]

    else:
        train = data.copy()
        test = pd.DataFrame(columns=['forecast_date', 'forecast'])

    train.to_csv(data_base_path+'/data_train.csv')
    test.to_csv(data_base_path+'/data_test.csv', index=False)

    return train
        

def delete_forecast_files():
    for filename in os.listdir(forecast_path):
        file_path = os.path.join(forecast_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return print("Previous forecast files deleted")
