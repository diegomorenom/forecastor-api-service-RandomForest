
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten, LSTM
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import pandas as pd
import numpy as np
from datetime import timedelta


path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))
data_path = str(parent_path) + "/forecastor-api-service/data_processing"
sys.path.append(data_path)
from data_modeling import scale_back_data, train_test_split, labels_pred


class NeuralNetworkLSTM():
    def __init__(self, data, scaler, parameters, forecast_days):
        self.data = data 
        self.scaler = scaler
        self.forecast_days = forecast_days
        self.parameters = parameters
        self.epochs = self.parameters['model_parameters']['epochs']
        self.batch_size = self.parameters['model_parameters']['batch_size']
        self.time_window = self.parameters['model_parameters']['time_window']
        self.train_split_perc = self.parameters['model_parameters']['train_split_perc']
        
    def create_ANNLSTM_model(self):
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, self.time_window )))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.summary()
        return model
    
    def fit_model(self):
        print("Training model")
        x_train, x_val, y_train, y_val = train_test_split(self.data, self.train_split_perc)
        model = self.create_ANNLSTM_model()
        model.fit(x_train,y_train,epochs=self.epochs,validation_data=(x_val,y_val),batch_size=self.batch_size)
        #_, accuracy = model.evaluate(x_val,y_val)
        return model

    def predict(self, nn_model):
        df_forecast = self.data.copy()
        columns = len(self.data.columns)
        last_row = list(self.data.values[-1].tolist())
        print('Making predictions')
        for d in range(self.forecast_days):
            features = labels_pred(last_row)
            features = np.array(features).reshape((1, 1, columns-1))
            prediction = nn_model.predict(features)
            print(prediction[0])
            del last_row[0] 
            last_row.append(prediction[0][0])
            forecast_date = df_forecast.index.max() + timedelta(days=1)
            df_forecast.loc[forecast_date] = last_row 
        df_pred = scale_back_data(df_forecast, self.scaler)
        df_pred = df_pred[df_pred.columns[-1]]
        return df_pred
