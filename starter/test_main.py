import pytest
from fastapi.testclient import TestClient

from starter.main import app

client = TestClient(app)


def test_get_greetings_at_root():
    """
    Test a valid request.
    Expected:
    - status code indicates success
    - response body matches the expected output format/content
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "hello there, this is an API which predicts salary based on the Census dataset"


def test_less_than_50k():
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
    assert r.json() == " <=50K"

def test_more_than_50k():
    
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
    assert r.json() == " >50K"

