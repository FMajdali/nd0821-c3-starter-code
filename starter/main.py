# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.testclient import TestClient
from pathlib import Path
import joblib
import numpy as np

current_script_dir = Path.cwd()
model_path = current_script_dir / "model" / "random_forest_model.joblib"
model = joblib.load(model_path)

encoder_path = current_script_dir / "model" / "encoder.joblib"
encoder = joblib.load(encoder_path)


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
app = FastAPI()

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

    data = item.model_dump()

    cat_lst = []
    num_lst = []
    for key,val in data.items():
        if key in cat_features:
            cat_lst.append(val)
        else:
            num_lst.append(val)
    

    encoded_arr = encoder.transform([cat_lst])
    print(f"encoded array {encoded_arr.shape}")
    num_arr = np.array([num_lst])
    print(f"num_arr array {num_arr.shape}")
    input_arr = np.concatenate([encoded_arr, num_arr], axis=1)
    print(input_arr.shape)
    print(f"and pred is {model.predict(input_arr)}")

    return item


client = TestClient(app)
r = client.get("/")
assert r.status_code == 200
print(r.json())

payload = {
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

r = client.post("/inference/",json=payload)
assert r.status_code == 200
print(r.json())