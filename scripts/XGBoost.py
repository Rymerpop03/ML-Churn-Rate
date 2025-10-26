'''
XGBoost Classification Model
Using XGBoost Classifier for churn prediction with the same preprocessing pipeline.
'''

import pandas as pd
import joblib as jb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

train_data = pd.read_csv('C:/Users/poke5/Desktop/Projects/ML-Churn/data/processed/train_data.csv')
test_data = pd.read_csv('C:/Users/poke5/Desktop/Projects/ML-Churn/data/processed/test_data.csv')

# Separate features and target variable
X_train = train_data.drop(['Churn', 'customerID'], axis=1)
y_train = (train_data['Churn'] == 'Yes').astype(int)
X_test = test_data.drop(['Churn', 'customerID'], axis=1)
y_test = (test_data['Churn'] == 'Yes').astype(int)

# Identify numeric and categorical columns
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features) 
    ])

# Create a pipeline with XGBoost
XGBModel = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    ))
])

# Train the model
print("Training XGBoost model...")
XGBModel.fit(X_train, y_train)

# Make predictions
y_pred = XGBModel.predict(X_test)

# Model Performance
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Accuracy
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")

# Save the model
model_path = 'C:/Users/poke5/Desktop/Projects/ML-Churn/results/models/xgboost_model.pkl'
jb.dump(XGBModel, model_path)
print(f"\nModel saved to: {model_path}")