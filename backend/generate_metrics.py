import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import json
import os

def load_test_data():
    """Load test data for evaluation"""
    # Load your test data here
    # This should match the format used during training
    test_data = pd.read_csv('datasets/test_data.csv')
    X_test = test_data[['shear_rate', 'concentration']]
    y_test = test_data['viscosity']
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate a model and return metrics"""
    y_pred = model.predict(X_test)
    return {
        'r2': float(r2_score(y_test, y_pred)),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'mse': float(mean_squared_error(y_test, y_pred))
    }

def main():
    # Load test data
    X_test, y_test = load_test_data()
    
    # Load models
    models = {
        'rf': joblib.load('models/rf_model.joblib'),
        'gb': joblib.load('models/gb_model.joblib'),
        'svm': joblib.load('models/svm_model.joblib'),
        'ann': joblib.load('models/ann_model.joblib')
    }
    
    # Evaluate each model
    metrics = {}
    for name, model in models.items():
        metrics[name] = evaluate_model(model, X_test, y_test)
    
    # Save metrics to file
    with open('models/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("Model metrics have been generated and saved to models/model_metrics.json")
    print("\nMetrics Summary:")
    for name, metric in metrics.items():
        print(f"\n{name.upper()}:")
        for metric_name, value in metric.items():
            print(f"  {metric_name}: {value:.4f}")

if __name__ == '__main__':
    main() 