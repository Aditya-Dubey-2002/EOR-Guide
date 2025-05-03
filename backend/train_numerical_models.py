import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os
import json

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

def train_and_save_numerical_models():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Load the data
    file_path = './datasets/HPAM_viscosity.xlsx'  # Update with your actual path
    data = pd.read_excel(file_path)

    # Extract shear rate
    shear_rate = data['Shear Rate'].values
    concentrations = [0.005, 0.01, 0.02]  # Update with actual concentration column names if different

    numerical_model_metrics = {}

    # Iterate over each concentration
    for conc in concentrations:
        viscosity = data[conc].values

        print(f"\nFitting models for Concentration {conc*100:.1f}%...")

        # Fit Power Law Model
        try:
            popt_power, _ = curve_fit(power_law, shear_rate, viscosity, maxfev=5000)
            viscosity_pred_power = power_law(shear_rate, *popt_power)
            r2_power, mse_power, mae_power, acc_power = calculate_metrics(viscosity, viscosity_pred_power)
            print(f"Power Law Parameters: k = {popt_power[0]:.6f}, n = {popt_power[1]:.6f}")
            numerical_model_metrics[f'{conc}_power_law'] = {
                'k': float(popt_power[0]),
                'n': float(popt_power[1]),
                'r2': float(r2_power),
                'mse': float(mse_power),
                'mae': float(mae_power),
                'accuracy': float(acc_power)
            }
        except Exception as e:
            print(f"Power Law fitting failed: {e}")
            numerical_model_metrics[f'{conc}_power_law'] = None

        # Fit Carreau-Yasuda Model
        try:
            init_params = [np.min(viscosity), np.max(viscosity), 1, 1, 0.5]
            bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])
            popt_carreau, _ = curve_fit(carreau_yasuda, shear_rate, viscosity, p0=init_params, bounds=bounds, maxfev=5000)
            viscosity_pred_carreau = carreau_yasuda(shear_rate, *popt_carreau)
            r2_carreau, mse_carreau, mae_carreau, acc_carreau = calculate_metrics(viscosity, viscosity_pred_carreau)
            print(f"Carreau-Yasuda Parameters: mu_inf = {popt_carreau[0]:.6f}, mu_0 = {popt_carreau[1]:.6f}, "
                  f"lambda = {popt_carreau[2]:.6f}, a = {popt_carreau[3]:.6f}, n = {popt_carreau[4]:.6f}")
            numerical_model_metrics[f'{conc}_carreau_yasuda'] = {
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
            numerical_model_metrics[f'{conc}_carreau_yasuda'] = None

    # Save numerical model parameters
    with open('models/numerical_model_parameters.json', 'w') as f:
        json.dump(numerical_model_metrics, f, indent=4)

    print("\nNumerical models trained and saved successfully!")

if __name__ == '__main__':
    train_and_save_numerical_models()