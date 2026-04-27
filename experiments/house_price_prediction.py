import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["SalePrice"])
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, preprocessor

def build_regression_model(preprocessor):
    model = RandomForestRegressor(random_state=42)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    return pipeline

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    metrics = pd.DataFrame({
        "MAE": [mae],
        "RMSE": [rmse],
        "R2 Score": [r2]
    })
    return predictions, metrics

def visualize_results(y_test, predictions):
    plt.figure()
    sns.scatterplot(x=y_test, y=predictions)
    plt.show()

X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data("datasets/HousePricePrediction.csv")
model = build_regression_model(preprocessor)
model.fit(X_train, y_train)
predictions, metrics = evaluate_model(model, X_test, y_test)
print(metrics)
visualize_results(y_test, predictions)
