# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
# Add the necessary imports for the starter code.
from starter.ml.data import process_data
from starter.ml.model import *
# Add code to load in the data.
current_script_dir = Path(__file__).resolve().parent
data_path = current_script_dir.parent / "data" / "census.csv"

data = pd.read_csv(data_path)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train,test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,encoder=encoder
)
# Train and save a model.
