from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import numpy as np
import os
import random

P = 0.7

app = FastAPI(port=5700)
mlflow_uri = os.getenv("MLFLOW_URI", "http://localhost:8080")
mlflow.set_tracking_uri(mlflow_uri)

curr_model_name = "gdp_logistic_regression"
curr_model_version = "latest"
model_uri = f"models:/{curr_model_name}/{curr_model_version}"

next_model_name = "gdp_logistic_regression"
next_model_version = "latest"

current = mlflow.sklearn.load_model(model_uri)
next = mlflow.sklearn.load_model(model_uri)

class GDPInput(BaseModel):
    year_2020: float
    year_2021: float
    year_2022: float
    year_2023: float
    year_2024: float

class ModelVersionRequest(BaseModel):
    model_name: str
    version: str

@app.get("/current-model")
def get_current_model():
    return {
        "name": curr_model_name,
        "version": curr_model_version
        }

@app.get("/canary-deployment-info")
def get_canary_deployment_info():
    return { 
        "current_model": {
            "name": curr_model_name,
            "version": curr_model_version
        },
        "next_model": {
            "name": next_model_name,
            "version": next_model_version
        },
        "P": P
    }

@app.get("/models")
def get_all_models():
    client = mlflow.tracking.MlflowClient()
    models = client.search_registered_models()
    model_names = [
        {
            "name": model.name,
            "versions": [
                v.version 
                for v in client.search_model_versions(f"name='{model.name}'")
                ]
         } 
        for model in models 
    ]

    return {"models": model_names}

@app.post("/predict")
def predict_gdp(data: GDPInput):
    input_data = np.array([[
            data.year_2020,
            data.year_2021,
            data.year_2022,
            data.year_2023,
            data.year_2024
        ]], dtype=np.float64)

    if random.random() < P:
        selected_model = current
        model_type = "current"
        model_name = curr_model_name
        model_version = curr_model_version
    else:
        selected_model = next
        model_type = "next"
        model_name = next_model_name
        model_version = next_model_version

    prediction = selected_model.predict(input_data).tolist()

    return {
        "y_pred": prediction[0],
        "model": f"{model_name}:{model_version}",
        "model_type": model_type
        }

@app.post("/update-model")
def update_model_version(request: ModelVersionRequest):
    global next, next_model_name, next_model_version
    next = mlflow.sklearn.load_model(f"models:/{request.model_name}/{request.version}")
    next_model_name = request.model_name
    next_model_version = request.version

    return {"message": f"Next model updated to: {request.model_name}:{request.version}"}

@app.post("/accept-next-model")
def accept_next_model():
    global current, curr_model_name, curr_model_version
    current = next
    curr_model_name = next_model_name
    curr_model_version = next_model_version

    return {"message": "Next model has been promoted to current model"}
