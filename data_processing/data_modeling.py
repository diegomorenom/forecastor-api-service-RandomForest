import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))
data_base_path = str(parent_path)+"/forecastor-api-service/data_processing/data_base"
forecast_path = str(parent_path)+"/forecastor-api-service/data_processing/forecast_files"

def model_data(df):
    print('Preparing data set')
    dates_list = list(df.index)
    scaled_data, scaler = scale_data(df)
    df_scaled = pd.DataFrame(scaled_data,index=(dates_list))
    df_modeled = series_to_supervised(df_scaled, 7, 1)
    return df_modeled, scaler

def scale_data(df):
    values = df.values
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    values=values.reshape(-1, 1) 
    scaled = scaler.fit_transform(values)
    return scaled, scaler

def scale_back_data(df, scaler):
    dates_list = list(df.index)
    new_df = scaler.inverse_transform(df)
    new_df = pd.DataFrame(new_df, index=(dates_list), columns=df.columns)
    return new_df

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    if dropnan: 
        agg.dropna(inplace=True)
    return agg

def train_test_split(data, split_train):
    
    train_dataset= 1-(1/split_train)

    model_data_long = data.shape
    model_data_long = model_data_long[0]

    n_train_days = round(model_data_long*train_dataset )
    train = data.iloc[:n_train_days, :]
    test = data.iloc[n_train_days:, :]

    x_train, y_train = np.array(train.iloc[:, :-1]), np.array(train.iloc[:, -1])
    x_val, y_val = np.array(test.iloc[:, :-1]), np.array(test.iloc[:, -1])

    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
    
    return x_train, x_val, y_train, y_val

def labels_pred(last_row):
    features_list = []
    columns = len(last_row)
    for c in range(columns):
        if c != 0:
            sales = last_row[c]
            features_list.append(sales)
    return features_list  

def get_error_metrics(models):
    df_real = pd.read_csv(data_base_path+"/data_test.csv")
    df_real = df_real.rename({'forecast': 'actual'}, axis='columns')
    model_error = {}
    for model in models:
        df_pred = pd.read_csv(forecast_path+"/forecast_"+model+".csv")
        df_pred = df_pred[['forecast_date','forecast']]
        df_pred.columns = ['forecast_date', model]
        df_error = df_real.merge(df_pred, how='left', on='forecast_date')
        model_error[model] = {}
        model_error[model]['MAE'] = mean_absolute_error(df_error['actual'], df_error[model]) 
        model_error[model]['MAPE'] = mean_absolute_percentage_error(df_error['actual'], df_error[model])
        model_error[model]['MSE'] = mean_squared_error(df_error['actual'], df_error[model]) 
        model_error[model]['RMSE'] = mean_squared_error(df_error['actual'], df_error[model], squared=False) 
        model_error[model]['R2'] = r2_score(df_error['actual'], df_error[model])
    return model_error