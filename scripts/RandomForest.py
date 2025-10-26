'''
Random Forest Classification Model
Using Random Forest Classifier for churn prediction with the same preprocessing pipeline.
'''

import pandas as pd
import joblib as jb
from sklearn.ensemble import RandomForestClassifier
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

# Create a pipeline with Random Forest
RFModel = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )) 
])

# Train the model
RFModel.fit(X_train, y_train)

# Make predictions
y_pred = RFModel.predict(X_test)

# Model Performance
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Accuracy
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")

# Save the model
jb.dump(RFModel, 'C:/Users/poke5/Desktop/Projects/ML-Churn/results/models/random_forest_model.pkl')