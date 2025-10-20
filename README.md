# ML-Churn-Rate (WIP)
I am stepping into the shoes of a data scientist working for a subscription-based company. I am tasked with building a predictive model that identifies customers who are likely to churn so our business can intervene early with personalzied offers/improved services. 

# Project Overview 
- My goal is to reduce churn by identifying at-risk customers. I will be building a machine learning model to predict customer churn with >85% recall while making sure to maintain balanced precision, and target proactive customer retention strategies. 
- KPIs: Churn Rate Reduction Percentage, Recall, Precision
- Tools Used: Tableau (Exploratory Data Analysis), Python/Jupyter Notebook (Feautre Engineering & Modeling)
- Dataset: Customer Churn dataset based on telephone service provider statistics. Dataset contains 21 features and 7043 records. 

# Data Description
Telco Customer Churn - https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download

Key Attributes
- customerID
- gender
- SeniorCitizen (Boolean)
- Partner
- tenure
- Contract
- PaymentMethod
- MonthlyCharges
- TotalCharges
- Churn

# Procedure
1. Problem Framing
2. Exploratory Data Analysis
3. Feature Engineering & Modelling
4. Model Evaluation & Business Recommendations

# Repository Structure
```
ML-Churn-Rate/
│-- data/
│   ├── raw/  
│   ├── processed/
│-- notebooks/
│   ├── 1_business_objective.ipynb
│   ├── 2_eda.ipynb
│   ├── 3_feature_engineering_modelling.ipynb
│   ├── 4_model_evaluation_and_recommendations.ipynb
│-- scripts/
│   ├── preprocess.py
│   ├── train_model.py
│-- README.md
│-- requirements.txt
│-- results/
│   ├── figures/
│   ├── model_performance.csv
```
