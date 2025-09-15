from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import numpy as np
import os

app = FastAPI(port=5700)
mlflow_uri = os.getenv("MLFLOW_URI", "http://localhost:8080")
mlflow.set_tracking_uri(mlflow_uri)

model_name = "gdp_logistic_regression"
model_version = "latest"
model_uri = f"models:/{model_name}/{model_version}"
mlflow_model = mlflow.sklearn.load_model(model_uri)

class GDPInput(BaseModel):
    year_2020: float
    year_2021: float
    year_2022: float
    year_2023: float
    year_2024: float

class ModelVersionRequest(BaseModel):
    model_name: str
    version: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the GDP Prediction API"}

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
    
    prediction = mlflow_model.predict(input_data).tolist()

    return {
        "y_pred": prediction[0],
        "model": f"{model_name}:{model_version}"
        }

@app.put("/update-model")
def update_model_version(request: ModelVersionRequest):
    global mlflow_model, model_name, model_version
    mlflow_model = mlflow.sklearn.load_model(f"models:/{request.model_name}/{request.version}")
    model_name = request.model_name
    model_version = request.version

    return {"message": f"Using model: {model_name}:{model_version}"}
