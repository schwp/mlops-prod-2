import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mlflow.set_tracking_uri("http://localhost:8080")

gdp_experiment = mlflow.set_experiment("GDP_Estimation_Models")
run_name = "GDP_LogisticRegression_Model"
artifact_path = "lr_gdp_model"

df = pd.read_csv("../data/gdp_per_country.csv")
df = df.dropna()
df_inputs = df[["2020", "2021", "2022", "2023", "2024"]].apply(pd.to_numeric, errors='coerce').dropna()
df_target = df["2025"].apply(pd.to_numeric, errors='coerce').dropna()

params = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 100000,
    "random_state": 42
}

X_train, X_test, y_train, y_test = train_test_split(df_inputs, df_target, test_size=0.2, random_state=42)

model = LogisticRegression(**params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params(model.get_params())
    mlflow.log_metrics(metrics)
    signature = infer_signature(X_train, y_pred)
    mlflow.sklearn.log_model(
        sk_model=model,
        name=artifact_path,
        input_example=X_test[:5],
        signature=signature,
        registered_model_name="gdp_logistic_regression"
    )
