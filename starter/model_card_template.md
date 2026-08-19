# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- Author: Faris Majdali
- Model date: 19-08-2026
- Model version: 1.0.0
- This model was devolped as part of the Udacity ML DevOps Engineer NanoDegree
- Model type: Random Forest Classifier
- The model was trained on the "Census" dataset to predict salary segment based on some features
- Detials about model fairness could be found in the slice_output.txt
- License: Copyright © 2012 - 2020, Udacity, Inc.

## Intended Use
This model is trained and devopled for learning purposes, not fit for production use

## Training Data
- 80% of the "Census" data set, the split was stratified on "salary"
- Pre-Processing: categorical data were one-hot-encoded, numerical data remained the same, labels were transformed into [0,1] labels
## Evaluation Data
- 20% of the "Census" data set, the split was stratified on "salary"
- Pre-Processing: categorical data were one-hot-encoded, numerical data remained the same, labels were transformed into [0,1] labels
## Metrics
- the metrics of the model on the test data are: 
- precision: 68.1%
- recall: 62.63%
- fbeta: 65.25%

## Ethical Considerations
- This model is trained and devopled for learning purposes, not fit for production use
## Caveats and Recommendations
- The model requires further hyper-parameter tunning
- Deeper dive on slices is required to assure model fairness