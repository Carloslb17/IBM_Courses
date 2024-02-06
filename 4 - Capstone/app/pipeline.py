import pandas as pd 
import numpy as np
import seaborn as sns

from ml_models import ml_models
import mlflow

from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt



class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, strategy='median'):
        super().__init__()
        self.columns = columns
        self.strategy = strategy  # Store strategy as an attribute
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X, y=None):
        # Fit the scaler and imputer to the numeric columns
        self.scaler.fit(X[self.columns])
        self.imputer.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()

        # Impute missing values
        X_transformed[self.columns] = self.imputer.transform(X[self.columns])

        # Scale numeric columns
        X_transformed[self.columns] = self.scaler.transform(X_transformed[self.columns])

        return X_transformed

def creation_pipeline(test_csv, train_csv, name):
    # Load and preprocess the data
    df_train = pd.read_csv(train_csv).dropna()
    df_test = pd.read_csv(test_csv).dropna()

    # Set up MLflow experiment
    mlflow.set_experiment(name)
    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("test_csv", test_csv)
        mlflow.log_param("train_csv", train_csv)
        mlflow.set_tag("tag_name", name)

        y_train = df_train[['Target']]
        y_test =  df_test[['Target']]

        X_train = df_train.drop('Target', axis=1)
        X_test = df_test.drop('Target', axis=1)

        # Extract numeric column names
        numeric_columns = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # Define the preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('scaler_imputer', CustomScaler(columns=numeric_columns), numeric_columns),
                ('scaler', StandardScaler(), numeric_columns),
            ],
            remainder='passthrough'
        )

        # Define classifiers
        classifiers = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier(),
            'Neural Network': MLPClassifier(max_iter=1000)
        }

        # Create a pipeline for each classifier
        pipelines = {}
        for clf_name, clf in classifiers.items():
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', clf)
            ])
            pipelines[clf_name] = pipeline

        # Set hyperparameters grid for GridSearchCV
        param_grids = {
            'Logistic Regression': {'classifier__C': [0.1, 1.0]},
            'Random Forest': {'classifier__n_estimators': [100, 200]},
            'Neural Network': {'classifier__hidden_layer_sizes': [(64,), (128,)]}
        }

        # Perform GridSearchCV to find the best model
        best_models = {}
        for clf_name, pipeline in pipelines.items():
            grid_search = GridSearchCV(pipeline, param_grid=param_grids[clf_name], cv=2)
            grid_search.fit(X_train, y_train)
            best_models[clf_name] = grid_search.best_estimator_

            # Log hyperparameters
            # Log metrics
            mlflow.log_metric(f"{clf_name}_accuracy", best_models[clf_name].score(X_train, y_train))

        # End the nested run
        mlflow.end_run()
