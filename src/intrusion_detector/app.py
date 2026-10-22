# Expose predictions through FastAPI

from fastapi import FastAPI, HTTPException
import joblib
from mangum import Mangum
import pandas as pd
import os
import time
import uuid  # NEW: gives every prediction a unique id

import boto3 # for dynamodb

# to validate structure of the incoming traffic
# so the api rejects malformed requests
from pydantic import BaseModel
from intrusion_detector.config import model_path


app = FastAPI(
    title = "Network instrusion Detection API",
    description = "API for detecting network attacks using a trained Random Forest model.",
    version = "1.0"

)

# wrap in mangum for aws lambda
handler = Mangum(app)

# Initialize DynamoDB client (uses environment variable for table name)
dynamodb = boto3.resource("dynamodb")
table_name = os.getenv("DYNAMODB_TABLE_NAME", "NIDS-Predictions")
table = dynamodb.Table(table_name)

# load trained model into memory once
try:
    model = joblib.load(model_path)
except Exception as e:
    model = None



# Define the data structure and validation rules for network traffic entering the API
class PredictionRequest(BaseModel):
    duration: float
    protocol_type: str
    service: str
    flag: str
    src_bytes: float
    dst_bytes: float
    land: float
    wrong_fragment: float
    urgent: float
    hot: float
    num_failed_logins: float
    logged_in: float
    num_compromised: float
    root_shell: float
    su_attempted: float
    num_root: float
    num_file_creations: float
    num_shells: float
    num_access_files: float
    num_outbound_cmds: float
    is_host_login: float
    is_guest_login: float
    count: float
    srv_count: float
    serror_rate: float
    srv_serror_rate: float
    rerror_rate: float
    srv_rerror_rate: float
    same_srv_rate: float
    diff_srv_rate: float
    srv_diff_host_rate: float
    dst_host_count: float
    dst_host_srv_count: float
    dst_host_same_srv_rate: float
    dst_host_diff_srv_rate: float
    dst_host_same_src_port_rate: float
    dst_host_srv_diff_host_rate: float
    dst_host_serror_rate: float
    dst_host_srv_serror_rate: float
    dst_host_rerror_rate: float
    dst_host_srv_rerror_rate: float



# creating endpoints

@app.get("/")
def home():
    return {"status": "online", "message": "NIDS API is running successfully."}

@app.post("/predict")
def predict_traffic(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model pipeline not loaded. Train the model first.")
    
    try:
        # Convert the validated Pydantic model into a dictionary, then to a DataFrame (1 row)
        input_data = request.model_dump()
        df = pd.DataFrame([input_data])
        
        # Make prediction using the pipeline (handles preprocessing automatically)
        prediction = model.predict(df)[0]
        
        # Map numerical prediction back to human-readable label
        result_label = "attack" if prediction == 1 else "normal"
        
        # Log prediction result to DynamoDB (wrapped in try/except so local testing doesn't crash if AWS creds aren't configured yet)
        # CHANGED: only a few fields are logged, and numbers are stored as strings
        # (DynamoDB does not accept Python floats)
        try:
            table.put_item(
                Item={
                    "prediction_id": str(uuid.uuid4()),
                    "prediction_code": int(prediction),
                    "prediction_label": result_label,
                    "protocol_type": request.protocol_type,
                    "service": request.service,
                    "src_bytes": str(request.src_bytes),
                    "dst_bytes": str(request.dst_bytes),
                }
            )
        except Exception:
            pass # Skips logging locally if AWS credentials aren't active yet

        return {
            "prediction_code": int(prediction),
            "prediction_label": result_label
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")