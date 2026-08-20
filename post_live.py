import requests

url = "https://nd0821-c3-starter-code-jfxl.onrender.com//inference"

data = {
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

response = requests.post(url, json=data)

print("Status code:", response.status_code)
print("Response:", response.json())
