# Confusion matrix and feature importance script
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import os
from sklearn.metrics import confusion_matrix

def plot_model_analysis(model_path, test_data_path, output_name):

    # Load the model
    model = joblib.load(model_path)
    
    # Load test data
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.drop(['Churn', 'customerID'], axis=1)
    y_test = (test_data['Churn'] == 'Yes').astype(int)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Create visualizations
    plt.figure(figsize=(15, 6))
    
    # Plot 1: Confusion Matrix
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Churned', 'Churned'],
                yticklabels=['Not Churned', 'Churned'])
    plt.title(f'Confusion Matrix - {output_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Plot 2: Feature Importance
    plt.subplot(1, 2, 2)
    
    # Identify numeric and categorical columns
    numeric_features = X_test.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_test.select_dtypes(include=['object']).columns
    
    # Get feature names after preprocessing
    feature_names = (numeric_features.tolist() +
                    [f"{feat}_{val}" for feat, vals in 
                     zip(categorical_features,
                         model.named_steps['preprocessor']
                         .named_transformers_['cat']
                         .categories_) 
                     for val in vals[1:]])
    
    # Get feature importance based on model type
    if hasattr(model.named_steps['classifier'], 'coef_'):
        importance_values = abs(model.named_steps['classifier'].coef_[0])
        importance_label = 'Coefficient'
    else:
        importance_values = model.named_steps['classifier'].feature_importances_
        importance_label = 'Importance'
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        importance_label: importance_values
    })
    
    # Sort by importance value and get top 10
    top_features = feature_importance.sort_values(importance_label, ascending=False).head(10)
    
    # Plot feature importance
    sns.barplot(data=top_features, y='Feature', x=importance_label, palette='viridis')
    plt.title(f'Top 10 Most Important Features - {output_name}')
    plt.xlabel(f'Feature {importance_label}')
    
    plt.tight_layout()
    
    # Save visualizations
    results_dir = 'C:/Users/poke5/Desktop/Projects/ML-Churn/results/visuals/'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save plot
    plt.savefig(os.path.join(results_dir, f'{output_name}_analysis.png'), 
                dpi=300, bbox_inches='tight')
    print(f"\nVisualizations saved to: {os.path.join(results_dir, f'{output_name}_analysis.png')}")
    
    # Display numerical feature importance
    print(f"\nTop 10 Most Important Features for {output_name}:")
    print(top_features.to_string(index=False))
    
    return feature_importance, cm

# Example usage:
if __name__ == "__main__":
    # Paths
    models_to_evaluate = {
        'logistic_regression': 'logistic_regression_model.pkl',
        'random_forest': 'random_forest_model.pkl',
        'xgboost': 'xgboost_model.pkl',
    }
    
    test_data_path = 'C:/Users/poke5/Desktop/Projects/ML-Churn/data/processed/test_data.csv'
    
    # Evaluate each model
    for model_name, model_file in models_to_evaluate.items():
        model_path = f'C:/Users/poke5/Desktop/Projects/ML-Churn/results/models/{model_file}'
        
        print(f"\nEvaluating {model_name}...")
        try:
            coefficients, cm = plot_model_analysis(
                model_path=model_path,
                test_data_path=test_data_path,
                output_name=model_name
            )
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")