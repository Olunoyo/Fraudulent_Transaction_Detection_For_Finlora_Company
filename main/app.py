from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path

from src.inference.prediction import (load_historical_data,
                                      predict_transaction,
                                      engineer_feature,
                                      add_transaction_to_cache,
                                      calculate_amount_usd)

from src.utility.model_loader import load_registered_model

app = FastAPI(title="fraud detection API")

model = None
load_historical_data = None


@app.on_event("startup")
async def startup_event():
    global model, historical_data
    try:
        model = load_registered_model()
        print("model loaded successfully")
    except Exception as e:
        print("error occurred while loading the model {e}")
        model = None

        try:
            #csv_path =r"C:/Users/HP/Downloads/Fraudulent_Transaction_Detection_For_Finlora_Company/Finlora_Dataset/FinLora_Customer_Transaction_Dataset.csv" 


            BASE_DIR = Path(__file__).resolve().parent.parent
            csv_path = BASE_DIR / "Finlora_Dataset" / "artifacts" / "Customer_Transaction_Dataset.csv"  
              
            historical_data = load_historical_data(csv_path)
            print("historical data has been successfully created")
        except Exception as e:
            print("error occourred during loading of historical dataset {e}")
            historical_data = None

## What the API is expecting from the user as an input
class TransactionData(BaseModel):
    timestamp: str
    customer_id: str
    home_country: str
    source_currency: str
    dest_currency: str
    channel: str
    amount_sc: float
    fee: float
    new_device: Optional[str] = "No"
    ip_country: str
    location_mismatch: Optional[str] = "No"
    ip_risk_score: float
    kyc_tier: str
    account_age_days: int 
    device_trust_score: float 
    risk_score_internal: float
    corridor_risk: float

# What the API is going to giving back to the user as an output or response
class PredictionResponse(BaseModel):
    is_fraud: int
    fraud_probability: float
    txn_velocity_1h: Optional[int] = None
    txn_velocity_24h: Optional[int] = None
    velocity_spike: Optional[int] = None
    amount_usd: Optional[float] = None

# creating the predict API so that our model can get users requests and make prediction
@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionData):
    global model, historical_data

    if model is None: 
         raise HTTPException(status_code = 500, detail="model not loaded")
     
    if historical_data is None:
         raise HTTPException(status_code = 500, detail = "historical data has not been loader")

   
     