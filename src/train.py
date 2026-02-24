import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import set_config
from config import *

set_config(transform_output="pandas")

def train_model(data=None):

    if data is None:
        train_df = pd.read_csv("data/train.csv")
    else:
        train_df = data

    test_df = pd.read_csv("data/test.csv")

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("feature_selection", SelectKBest(score_func=f_classif, k=200)),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=1
        ))
    ])

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("feature_selection_k", 200)

        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
        mlflow.log_metric("test_accuracy", test_accuracy)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME
        )

        print("CV Accuracy:", cv_scores.mean())
        print("Test Accuracy:", test_accuracy)

if __name__ == "__main__":
    train_model()
