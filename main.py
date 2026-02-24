# ==========================================================
# HUMAN ACTIVITY RECOGNITION - RESEARCH GRADE PIPELINE
# ==========================================================

import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import set_config
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# 1. CONFIGURATION (Best Practice)
# ==========================================================

# Ensures all transformers return pandas DataFrame
set_config(transform_output="pandas")

RANDOM_STATE = 42
N_SPLITS = 5

# ==========================================================
# 2. LOAD DATA
# ==========================================================

train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

# Assuming 'Activity' is target column
TARGET_COLUMN = "Activity"

X_train = train_df.drop(columns=[TARGET_COLUMN])
y_train = train_df[TARGET_COLUMN]

X_test = test_df.drop(columns=[TARGET_COLUMN])
y_test = test_df[TARGET_COLUMN]

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ==========================================================
# 3. BUILD PIPELINE
# ==========================================================

pipeline = Pipeline([
    
    # Standardization
    ("scaler", StandardScaler()),
    
    # Feature selection (Top 200 best features)
    ("feature_selection", SelectKBest(score_func=f_classif, k=200)),
    
    # Classifier
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=1
    ))
])

# ==========================================================
# 4. CROSS VALIDATION
# ==========================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=1
)

print("\nCross Validation Results")
print("------------------------")
print("Fold Accuracies:", cv_scores)
print("Mean CV Accuracy: {:.4f}".format(cv_scores.mean()))

# ==========================================================
# 5 & 6. TRAIN + EVALUATE + MLFLOW TRACKING
# ==========================================================

import mlflow
import mlflow.sklearn

mlflow.set_experiment("HAR_Experiment")

with mlflow.start_run():

    # Train model
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)

    print("\nTest Set Performance")
    print("------------------------")
    print("Test Accuracy: {:.4f}".format(test_accuracy))

    print("\nClassification Report")
    print("------------------------")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix")
    print("------------------------")
    print(confusion_matrix(y_test, y_pred))

    # Log parameters
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("feature_selection_k", 200)
    mlflow.log_param("cv_folds", N_SPLITS)

    # Log metrics
    mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
    mlflow.log_metric("test_accuracy", test_accuracy)

    # Log & Register model
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name="HAR_Model"
    )

print("\n✅ Model logged & registered in MLflow")

print("\n✅ Model saved successfully as models/har_model_v1.pkl")
