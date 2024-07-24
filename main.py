from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
from forecast_generator import forecast_process as fp

import json
import os
import sys
import string

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))
data_path = str(parent_path) + "/forecastor-api-service/data_processing"
forecast_path = str(parent_path) + "/forecastor-api-service/forecast_generator"
store_path = str(parent_path) + "/forecastor-api-service/data_processing/data_base"
forecast_files = str(parent_path) + "/forecastor-api-service/data_processing/forecast_files"
parameters_file = str(parent_path) + "/forecastor-api-service/parameters.JSON" 
forecast_info_file = str(parent_path) + "/forecastor-api-service/forecast_info.JSON"

sys.path.append(data_path)
sys.path.append(forecast_path)

from data_handler import get_data
from data_modeling import get_error_metrics

forecastingType = 'realtimeForecasting'

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Forecastor"}

# Configure CORS settings
origins = [
    "http://localhost",
    "http://localhost:5174/",  # Add your React frontend URL here
]


#Allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastRequest(BaseModel):
    prediction_column: str = Field(..., description="Name of prediction column")
    date_column: str = Field(..., description="Name of date column")
    forecastDays: int = Field(..., description="Number of forecast days")
    selectedModels: List[str] = Field(..., description="List of selected models")
    #forecastingType: List[str] = Field(..., description="Forecasting type")

@app.post("/process_forecast")
async def process_forecast(prediction_column: str = Form(...),
                           date_column: str = Form(...),
                           forecast_days: int = Form(...),
                           selected_models: str = Form(...),
                           #forecasting_type: str = Form(...),
                           csv_file: UploadFile = File(...)):
    
    
    # SAVE FILE CODE

    # Read CSV file
    data = pd.read_csv(csv_file.file)
    
    data_columns = [str(date_column), str(prediction_column)]
    data = data[data_columns]
    data.columns = ['date', 'prediction_column']
    
    # Save the CSV file
    data.to_csv(store_path+'/data_api.csv')


    # FORECAST PROCESS CODE   
    selected_models = selected_models.translate({ord(c): None for c in string.whitespace})
    selected_models_list = selected_models.split(',')
    print(f"Received forecast days: {forecast_days}")
    print(f"Received selected models: {selected_models_list}")
    #print(f"Received forecasting type: {forecasting_type}")

    # Read CSV file
    print(parameters_file)
    parameters_json = open(parameters_file)
    parameters = json.load(parameters_json)

    # Instantiate ForecastingProcess class
    forecast_process_instance = fp.ForecastingProcess(data, selected_models_list, parameters[1], forecast_days, forecastingType)

    # Call run_all_models method
    forecast_process_instance.run_all_models()

    # Path to the generated forecast file
    forecast_output_file = forecast_files + '/forecast.csv'
    
    # Ensure the forecast output file exists before returning it
    if not os.path.exists(forecast_output_file):
        return {"message": "Forecast file was not generated."}

    return FileResponse(forecast_output_file, media_type='text/csv', filename='forecast.csv')

