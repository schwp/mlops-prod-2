import requests

url = "http://localhost:5700/predict"

def test_get_models():
    res = requests.get("http://localhost:5700/models")

    assert res.status_code == 200

    data = res.json()

    assert "models" in data
    assert isinstance(data["models"], list)

def test_predict():
    payload = {
        "year_2020": 1000.0,
        "year_2021": 1100.0,
        "year_2022": 1200.0,
        "year_2023": 1300.0,
        "year_2024": 1400.0
    }
    res = requests.post(url, json=payload)

    assert res.status_code == 200

    data = res.json()

    assert "y_pred" in data
    assert isinstance(data["y_pred"], float)
