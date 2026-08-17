# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import *
# Add code to load in the data.
current_script_dir = Path.cwd()
data_path = current_script_dir.parent / "data" / "census.csv"

data = pd.read_csv(data_path)
data.columns = [col.strip() for col in data.columns] 
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train,test = train_test_split(data, test_size=0.20, random_state=42, stratify = data['salary'])

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
X_test, y_test, encoder , _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,encoder=encoder
)
# Train and save a model.
model = train_model(X_train, y_train)
joblib.dump(model, current_script_dir.parent / "model" / "random_forest_model.joblib")
joblib.dump(encoder, current_script_dir.parent / "model" / "encoder.joblib")
joblib.dump(lb, current_script_dir.parent / "model" / "lb.joblib")

slice_eval(
        data_cols = data.columns,
        cat_features = cat_features,
        label = 'salary',
        x = X_test,
        y = y_test,
        slice_feature = 'education',
        model = model,
        encoder = encoder,
        lb = lb,
        output_file_name = "slice_output.txt"
    )