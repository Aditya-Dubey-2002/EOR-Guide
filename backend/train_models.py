import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from scipy.optimize import curve_fit  # Import curve_fit
import matplotlib.pyplot as plt  # Import matplotlib

def calculate_accuracy_within_20_percent(y_true, y_pred):
    """Calculate percentage of predictions within 20% of actual values"""
    within_20_percent = np.abs(y_true - y_pred) <= 0.2 * np.abs(y_true)
    return np.mean(within_20_percent) * 100

# Power Law Model: mu = k * (shear_rate)^(n-1)
def power_law(shear_rate, k, n):
    return k * shear_rate**(n-1)

# Carreau-Yasuda Model: mu = mu_inf + (mu_0 - mu_inf) * (1 + (lambda * shear_rate)^a )^((n-1)/a)
def carreau_yasuda(shear_rate, mu_inf, mu_0, lamb, a, n):
    return mu_inf + (mu_0 - mu_inf) * (1 + (lamb * shear_rate)**a )**((n-1)/a)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    accuracy = np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, 1e-8) <= 0.2) * 100  # Avoid zero-division
    return r2, mse, mae, accuracy

def train_and_save_models():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load the data
    file_path = './datasets/HPAM_viscosity.xlsx'  # Adjust the file path as needed
    data = pd.read_excel(file_path)
    
    # Extract features and targets
    shear_rate = data['Shear Rate'].values
    viscosity = data[[0.005, 0.01, 0.02]].values  # Concentrations: 0.5%, 1.0%, 1.5%
    
    # Apply log transformation
    shear_rate = np.log(shear_rate)
    viscosity = np.log(viscosity)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        shear_rate.reshape(-1, 1), 
        viscosity, 
        test_size=0.2, 
        random_state=42
    )
    
    # Scale the data for ANN and SVM
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Dictionary to store model metrics
    model_metrics = {}
    
    # Train Random Forest model
    print("Training Random Forest model...")
    rf_model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate Random Forest
    rf_predictions = rf_model.predict(X_test)
    rf_predictions_original = np.exp(rf_predictions)
    y_test_original = np.exp(y_test)
    
    model_metrics['rf'] = {
        'r2': float(r2_score(y_test, rf_predictions)),
        'mae': float(mean_absolute_error(y_test_original, rf_predictions_original)),
        'mse': float(mean_squared_error(y_test_original, rf_predictions_original))
    }
    print(f"Random Forest Metrics:")
    print(f"R² Score: {model_metrics['rf']['r2']:.4f}")
    print(f"MAE: {model_metrics['rf']['mae']:.4f}")
    print(f"MSE: {model_metrics['rf']['mse']:.4f}")
    
    # Train Gradient Boosting models for each concentration
    print("\nTraining Gradient Boosting models...")
    gb_models = []
    gb_metrics = {'r2': [], 'mae': [], 'mse': []}
    
    for i, concentration in enumerate([0.005, 0.01, 0.02]):
        print(f"Training GB model for {concentration*100}% concentration...")
        gb_model = GradientBoostingRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=3,
            min_samples_leaf=1,
            random_state=0
        )
        gb_model.fit(X_train, y_train[:, i])
        gb_models.append(gb_model)
        
        # Evaluate GB model
        gb_predictions = gb_model.predict(X_test)
        gb_predictions_original = np.exp(gb_predictions)
        y_test_original = np.exp(y_test[:, i])
        
        gb_metrics['r2'].append(float(r2_score(y_test[:, i], gb_predictions)))
        gb_metrics['mae'].append(float(mean_absolute_error(y_test_original, gb_predictions_original)))
        gb_metrics['mse'].append(float(mean_squared_error(y_test_original, gb_predictions_original)))
    
    # Calculate average metrics for GB
    model_metrics['gb'] = {
        'r2': float(np.mean(gb_metrics['r2'])),
        'mae': float(np.mean(gb_metrics['mae'])),
        'mse': float(np.mean(gb_metrics['mse']))
    }
    print("\nGradient Boosting Metrics (Average):")
    print(f"R² Score: {model_metrics['gb']['r2']:.4f}")
    print(f"MAE: {model_metrics['gb']['mae']:.4f}")
    print(f"MSE: {model_metrics['gb']['mse']:.4f}")
    
    # Train SVM models for each concentration
    print("\nTraining SVM models...")
    svm_models = []
    svm_metrics = {'r2': [], 'mae': [], 'mse': []}
    
    for i, concentration in enumerate([0.005, 0.01, 0.02]):
        print(f"Training SVM model for {concentration*100}% concentration...")
        svm_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)  # Using same parameters as notebook
        svm_model.fit(X_train_scaled, y_train_scaled[:, i])
        svm_models.append(svm_model)
        
        # Evaluate SVM model
        svm_predictions = svm_model.predict(X_test_scaled)
        
        # Create a full prediction array with zeros for other concentrations
        full_predictions = np.zeros((len(svm_predictions), 3))
        full_predictions[:, i] = svm_predictions
        
        # Inverse transform the predictions
        svm_predictions_original = np.exp(scaler_y.inverse_transform(full_predictions))
        y_test_original = np.exp(scaler_y.inverse_transform(y_test_scaled))
        
        svm_metrics['r2'].append(float(r2_score(y_test_scaled[:, i], svm_predictions)))
        svm_metrics['mae'].append(float(mean_absolute_error(y_test_original[:, i], svm_predictions_original[:, i])))
        svm_metrics['mse'].append(float(mean_squared_error(y_test_original[:, i], svm_predictions_original[:, i])))
    
    # Calculate average metrics for SVM
    model_metrics['svm'] = {
        'r2': float(np.mean(svm_metrics['r2'])),
        'mae': float(np.mean(svm_metrics['mae'])),
        'mse': float(np.mean(svm_metrics['mse']))
    }
    print("\nSVM Metrics (Average):")
    print(f"R² Score: {model_metrics['svm']['r2']:.4f}")
    print(f"MAE: {model_metrics['svm']['mae']:.4f}")
    print(f"MSE: {model_metrics['svm']['mse']:.4f}")
    
    # Train ANN model with TensorFlow
    print("\nTraining ANN model...")
    ann_model = Sequential([
        Dense(units=128, input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        tf.keras.layers.Activation('tanh'),
        Dropout(0.3),
        
        Dense(units=64),
        BatchNormalization(),
        tf.keras.layers.Activation('tanh'),
        Dropout(0.3),
        
        Dense(units=32),
        BatchNormalization(),
        tf.keras.layers.Activation('tanh'),
        Dropout(0.3),
        
        Dense(units=3)  # Output layer for 3 viscosities
    ])
    
    # Compile the model
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    ann_model.compile(optimizer=optimizer, loss='mse')
    
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    
    # Train the model
    ann_model.fit(
        X_train_scaled, 
        y_train_scaled,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate ANN model
    ann_predictions = ann_model.predict(X_test_scaled)
    ann_predictions_original = np.exp(scaler_y.inverse_transform(ann_predictions))
    y_test_original = np.exp(scaler_y.inverse_transform(y_test_scaled))
    
    model_metrics['ann'] = {
        'r2': float(r2_score(y_test_scaled, ann_predictions)),
        'mae': float(mean_absolute_error(y_test_original, ann_predictions_original)),
        'mse': float(mean_squared_error(y_test_original, ann_predictions_original))
    }
    print("\nANN Metrics:")
    print(f"R² Score: {model_metrics['ann']['r2']:.4f}")
    print(f"MAE: {model_metrics['ann']['mae']:.4f}")
    print(f"MSE: {model_metrics['ann']['mse']:.4f}")

    # Train Numerical Models
    numerical_model_metrics = {}

    for i, concentration in enumerate([0.005, 0.01, 0.02]):
        print(f"\nFitting Numerical models for Concentration {concentration*100:.1f}%...")
        viscosity_train = y_train[:, i]
        viscosity_test = y_test[:, i]
        shear_rate_train = X_train.flatten()  # Flatten X_train
        shear_rate_test = X_test.flatten()    # Flatten X_test

        # Fit Power Law Model
        try:
            popt_power, _ = curve_fit(power_law, shear_rate_train, viscosity_train, maxfev=5000)
            viscosity_pred_power = power_law(shear_rate_test, *popt_power)
            r2_power, mse_power, mae_power, acc_power = calculate_metrics(viscosity_test, viscosity_pred_power)
            print(f"Power Law Parameters: k = {popt_power[0]:.6f}, n = {popt_power[1]:.6f}")
            numerical_model_metrics[f'{concentration}_power_law'] = {
                'k': float(popt_power[0]),
                'n': float(popt_power[1]),
                'r2': float(r2_power),
                'mse': float(mse_power),
                'mae': float(mae_power),
                'accuracy': float(acc_power)
            }
        except Exception as e:
            print(f"Power Law fitting failed: {e}")
            numerical_model_metrics[f'{concentration}_power_law'] = None

        # Fit Carreau-Yasuda Model
        try:
            init_params = [np.min(viscosity_train), np.max(viscosity_train), 1, 1, 0.5]
            bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])
            popt_carreau, _ = curve_fit(carreau_yasuda, shear_rate_train, viscosity_train, p0=init_params, bounds=bounds, maxfev=5000)
            viscosity_pred_carreau = carreau_yasuda(shear_rate_test, *popt_carreau)
            r2_carreau, mse_carreau, mae_carreau, acc_carreau = calculate_metrics(viscosity_test, viscosity_pred_carreau)
            print(f"Carreau-Yasuda Parameters: mu_inf = {popt_carreau[0]:.6f}, mu_0 = {popt_carreau[1]:.6f}, "
                  f"lambda = {popt_carreau[2]:.6f}, a = {popt_carreau[3]:.6f}, n = {popt_carreau[4]:.6f}")
            numerical_model_metrics[f'{concentration}_carreau_yasuda'] = {
                'mu_inf': float(popt_carreau[0]),
                'mu_0': float(popt_carreau[1]),
                'lamb': float(popt_carreau[2]),
                'a': float(popt_carreau[3]),
                'n': float(popt_carreau[4]),
                'r2': float(r2_carreau),
                'mse': float(mse_carreau),
                'mae': float(mae_carreau),
                'accuracy': float(acc_carreau)
            }
        except Exception as e:
            print(f"Carreau-Yasuda fitting failed: {e}")
            numerical_model_metrics[f'{concentration}_carreau_yasuda'] = None

    model_metrics['numerical'] = numerical_model_metrics

    # Save the models and scalers
    print("\nSaving models and scalers...")
    joblib.dump(rf_model, 'models/rf_model.joblib')
    joblib.dump(gb_models, 'models/gb_model.joblib')
    joblib.dump(svm_models, 'models/svm_model.joblib')
    ann_model.save('models/ann_model.h5')  # Save TensorFlow model
    joblib.dump(scaler_X, 'models/scaler_X.joblib')
    joblib.dump(scaler_y, 'models/scaler_y.joblib')

    # Save numerical model parameters
    with open('models/numerical_model_parameters.json', 'w') as f:
        json.dump(numerical_model_metrics, f, indent=4)
    
    # Save model metrics
    # with open('models/model_metrics.json', 'w') as f:
    #     json.dump(model_metrics, f, indent=4)
    
    print("\nModels trained and saved successfully!")

if __name__ == '__main__':
    train_and_save_models()