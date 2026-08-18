import pytest
from sklearn.ensemble import RandomForestClassifier
from .ml.model import *
import joblib
from pathlib import Path
import numpy as np

current_script_dir = Path.cwd()
model_path = current_script_dir.parent / "model" / "random_forest_model.joblib"
model = joblib.load(model_path)


def test_train_model():
    x = [[0],[1],[0],[1],[0]]
    y = [1,1,1,1,1]
    
    rf_model = train_model(x,y)

    assert isinstance(rf_model, RandomForestClassifier)

def test_compute_model_metrics():
    y = [0,1,1,0,1]
    y_pred = [1,1,0,0,0]
    precision, recall, fbeta = compute_model_metrics(y,y_pred)

    assert type(precision) == float
    assert type(recall) == float
    assert type(fbeta) == float


def test_inference():
    X = [[1]*108]
    y_pred = inference(model, X)

    assert isinstance(y_pred, np.ndarray)
