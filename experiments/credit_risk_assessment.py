import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from xgboost import XGBClassifier

data = pd.read_csv("datasets/credit_risk_dataset.csv")

data = data[data['person_age'] <= 100]
data = data[data['person_emp_length'] <= 50]
data = data[data['person_income'] <= 3000000]

y = data['loan_status']
X = data.drop(['loan_status'], axis=1)

X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

categorical_cols = [
    cname for cname in X_train_full.columns
    if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == 'object'
]

numerical_cols = [
    cname for cname in X_train_full.columns
    if X_train_full[cname].dtype in ['int64', 'float64']
]

my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

numerical_transformer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('nums', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

model = XGBClassifier(
    n_estimators=1000,
    max_depth=3,
    learning_rate=0.05,
    objective='binary:logistic'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_valid)

print(metrics.classification_report(y_valid, preds))

output = pd.DataFrame({
    'Id': X_valid.index,
    'Loan Status': preds
})

output.to_csv("credit_risk_assessment.csv", index=False)