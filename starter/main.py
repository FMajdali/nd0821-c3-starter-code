# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
from starter.ml.data import process_data
from fastapi.testclient import TestClient
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

model = joblib.load(MODEL_DIR / "random_forest_model.joblib")
encoder = joblib.load(MODEL_DIR / "encoder.joblib")
lb = joblib.load(MODEL_DIR / "lb.joblib")

class Item(BaseModel):
    #name: str
    #description: str | None = None
    #price: float
    #tax: float | None = None
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    'age': 39,
                    'workclass': ' State-gov',
                    'fnlgt': 77516,
                    'education': ' Bachelors',
                    'education_num': 13,
                    'marital_status': ' Never-married',
                    'occupation': ' Adm-clerical',
                    'relationship': ' Not-in-family',
                    'race': ' White',
                    'sex': ' Male',
                    'capital_gain': 2174,
                    'capital_loss': 0,
                    'hours_per_week': 40,
                    'native_country': ' United-States',
                }
            ]
        }
    }
app = FastAPI(root_path="/proxy/8000")

@app.get("/")
async def greetings():
    return "hello there, this is an API which predicts salary based on the Census dataset"


@app.post("/inference/")
async def model_infer(item: Item):

    #print(list(item.model_dump().values()))

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
        ]

    arr,_,_,_ = process_data(
                            pd.DataFrame([item.model_dump()]), 
                            categorical_features=cat_features,
                            label=None,
                            training=False,
                            encoder=encoder,
                            lb=lb
                            )

    pred = model.predict(arr)
    return lb.inverse_transform(pred)[0]

if __name__ == "__main__":
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    print(r.json())

    payload = {
                'age': 45,
                'workclass': ' Federal-gov',
                'fnlgt': 232997,
                'education': ' Some-college',
                'education_num': 10,
                'marital_status': ' Married-civ-spouse',
                'occupation': ' Transport-moving',
                'relationship': ' Husband',
                'race': ' White',
                'sex': ' Male',
                'capital_gain': 0,
                'capital_loss': 0,
                'hours_per_week': 65,
                'native_country': ' United-States',
                }

    r = client.post("/inference/",json=payload)
    assert r.status_code == 200
    print(r.json())