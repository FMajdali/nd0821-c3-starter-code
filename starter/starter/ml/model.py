from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
import pandas as pd

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    rf_model = RandomForestClassifier(n_estimators=5, random_state=42)
    rf_model.fit(X_train, y_train)

    return rf_model

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    y_pred = model.predict(X)

    return y_pred

def slice_eval(
        data_cols,
        cat_features,
        label,
        x,
        y,
        slice_feature,
        model,
        encoder,
        lb,
        output_file_name = "slice_output.txt"
    ):
    output_lst = []
    
    num_features = [col for col in data_cols if col not in cat_features]
    num_features = [col for col in num_features if col != label]
    
    encoded_df = pd.DataFrame(
        x[:,len(num_features):],
        columns=encoder.get_feature_names_out(cat_features)
    )

    num_df = pd.DataFrame(
        x[:,:len(num_features)],
        columns = num_features
    )

    df = pd.concat([num_df,encoded_df],axis = 1)
    df[label] = y

    feature_cols = [col for col in df.columns if col.split("_")[0] == slice_feature]
    output_lst = [f"slice metrics using test data for the feature '{slice_feature}':\n\n"]

    for col in feature_cols:
        temp_df = df[df[col] == 1].reset_index(drop = True).copy()
        y = temp_df.pop(label)
        y_preds = model.predict(temp_df.values)

        # number of total rows
        total = y.shape[0]
        y_not_null = y[~y.isna()]
        
        # number of not null values
        not_null_num = y_not_null.shape[0]

        # remove null labels
        y_preds = y_preds[~y.isna()]
        y = y[~y.isna()]

        assert y_preds.shape[0] == y.shape[0]
        precision, recall, fbeta = compute_model_metrics(lb.transform(y),y_preds)
        output_lst.append(f"the metrics for the slice {col} are:\ntotal rows: {total} \nnot null rows: {not_null_num}\nprecision: {precision}\nrecall: {recall}\nfbeta: {fbeta}\n\n")
    
    with open("slice_output.txt", "w") as f:
        f.writelines(output_lst)