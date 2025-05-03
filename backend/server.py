from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# Load the trained models and scalers
def load_models():
    try:
        rf_model = joblib.load('models/rf_model.joblib')
        gb_model = joblib.load('models/gb_model.joblib')
        svm_model = joblib.load('models/svm_model.joblib')
        ann_model = load_model('models/ann_model.h5')
        scaler_X = joblib.load('models/scaler_X.joblib')
        scaler_y = joblib.load('models/scaler_y.joblib')
        return rf_model, gb_model, svm_model, ann_model, scaler_X, scaler_y
    except FileNotFoundError:
        print("Models not found. Please train and save the models first.")
        return None, None, None, None, None, None

# Initialize models and scalers
rf_model, gb_model, svm_model, ann_model, scaler_X, scaler_y = load_models()

# Valid concentration values
VALID_CONCENTRATIONS = [0.5, 1.0, 2.0]

# Model metadata
MODEL_METADATA = {
    'rf': {
        'name': 'Random Forest',
        'version': '1.0',
        'description': 'Multi-output Random Forest model for viscosity prediction',
        'input_range': {
            'shear_rate': {'min': 0.001, 'max': 1000},
            'concentration': {'valid_values': VALID_CONCENTRATIONS}
        }
    },
    'gb': {
        'name': 'Gradient Boosting',
        'version': '1.0',
        'description': 'Gradient Boosting models for viscosity prediction',
        'input_range': {
            'shear_rate': {'min': 0.001, 'max': 1000},
            'concentration': {'valid_values': VALID_CONCENTRATIONS}
        }
    },
    'svm': {
        'name': 'Support Vector Machine',
        'version': '1.0',
        'description': 'SVM models for viscosity prediction',
        'input_range': {
            'shear_rate': {'min': 0.001, 'max': 1000},
            'concentration': {'valid_values': VALID_CONCENTRATIONS}
        }
    },
    'ann': {
        'name': 'Artificial Neural Network',
        'version': '2.0',
        'description': 'Deep Neural Network with batch normalization and dropout for viscosity prediction',
        'architecture': '128-64-32-3 with tanh activation and dropout',
        'input_range': {
            'shear_rate': {'min': 0.001, 'max': 1000},
            'concentration': {'valid_values': VALID_CONCENTRATIONS}
        }
    }
}

def validate_input(shear_rate, concentration):
    """Validate input parameters"""
    errors = []
    
    # Validate shear rate
    if not isinstance(shear_rate, (int, float)):
        errors.append("Shear rate must be a number")
    elif shear_rate < MODEL_METADATA['rf']['input_range']['shear_rate']['min']:
        errors.append(f"Shear rate must be >= {MODEL_METADATA['rf']['input_range']['shear_rate']['min']}")
    elif shear_rate > MODEL_METADATA['rf']['input_range']['shear_rate']['max']:
        errors.append(f"Shear rate must be <= {MODEL_METADATA['rf']['input_range']['shear_rate']['max']}")
    
    # Validate concentration
    if not isinstance(concentration, (int, float)):
        errors.append("Concentration must be a number")
    elif concentration not in VALID_CONCENTRATIONS:
        errors.append(f"Concentration must be one of {VALID_CONCENTRATIONS}")
    
    return errors

def preprocess_input(shear_rate):
    """Preprocess input data for ANN and SVM models"""
    # Apply log transformation
    shear_rate_log = np.log(shear_rate)
    # Scale the data
    shear_rate_scaled = scaler_X.transform([[shear_rate_log]])
    return shear_rate_log, shear_rate_scaled

def postprocess_output(prediction):
    """Convert log-transformed predictions back to original scale"""
    # Inverse transform the scaled predictions
    prediction_original = scaler_y.inverse_transform(prediction)
    # Apply inverse log transformation
    return np.exp(prediction_original)

@app.route('/predict/rf', methods=['POST'])
def predict_rf():
    """
    Endpoint for Random Forest predictions
    Expected input format:
    {
        "shear_rate": 0.001,
        "concentration": 0.5,
        "request_id": "optional_request_id"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No input data provided',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Extract and validate input
        shear_rate = data.get('shear_rate')
        concentration = data.get('concentration')
        request_id = data.get('request_id', '')
        
        errors = validate_input(shear_rate, concentration)
        if errors:
            return jsonify({
                'status': 'error',
                'message': 'Invalid input parameters',
                'errors': errors,
                'timestamp': datetime.now().isoformat()
            }), 400

        # Preprocess input
        shear_rate_log, _ = preprocess_input(shear_rate)

        # Make prediction
        X = np.array([[shear_rate_log]])
        predictions = rf_model.predict(X)
        # Convert predictions back to original scale
        predictions = postprocess_output(predictions)
        viscosity = float(predictions[0][VALID_CONCENTRATIONS.index(concentration)])
        
        # Format response
        response = {
            'status': 'success',
            'data': {
                'shear_rate': shear_rate,
                'concentration': concentration,
                'viscosity': viscosity,
                'model': 'Random Forest'
            },
            'metadata': {
                'request_id': request_id,
                'timestamp': datetime.now().isoformat(),
                'model_version': MODEL_METADATA['rf']['version']
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/gb', methods=['POST'])
def predict_gb():
    """
    Endpoint for Gradient Boosting predictions
    Expected input format:
    {
        "shear_rate": 0.001,
        "concentration": 0.5,
        "request_id": "optional_request_id"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No input data provided',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Extract and validate input
        shear_rate = data.get('shear_rate')
        concentration = data.get('concentration')
        request_id = data.get('request_id', '')
        
        errors = validate_input(shear_rate, concentration)
        if errors:
            return jsonify({
                'status': 'error',
                'message': 'Invalid input parameters',
                'errors': errors,
                'timestamp': datetime.now().isoformat()
            }), 400

        # Preprocess input
        shear_rate_log = np.log(shear_rate)
        X = np.array([[shear_rate_log]])

        # Make prediction
        prediction = gb_model[VALID_CONCENTRATIONS.index(concentration)].predict(X)
        
        # Convert prediction back to original scale
        prediction = np.exp(prediction)
        viscosity = float(prediction[0])
        
        # Format response
        response = {
            'status': 'success',
            'data': {
                'shear_rate': shear_rate,
                'concentration': concentration,
                'viscosity': viscosity,
                'model': 'Gradient Boosting'
            },
            'metadata': {
                'request_id': request_id,
                'timestamp': datetime.now().isoformat(),
                'model_version': MODEL_METADATA['gb']['version']
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/svm', methods=['POST'])
def predict_svm():
    """
    Endpoint for SVM predictions
    Expected input format:
    {
        "shear_rate": 0.001,
        "concentration": 0.5,
        "request_id": "optional_request_id"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No input data provided',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Extract and validate input
        shear_rate = data.get('shear_rate')
        concentration = data.get('concentration')
        request_id = data.get('request_id', '')
        
        errors = validate_input(shear_rate, concentration)
        if errors:
            return jsonify({
                'status': 'error',
                'message': 'Invalid input parameters',
                'errors': errors,
                'timestamp': datetime.now().isoformat()
            }), 400

        # Preprocess input
        shear_rate_log = np.log(shear_rate)
        shear_rate_scaled = scaler_X.transform([[shear_rate_log]])

        # Make prediction
        prediction = svm_model[VALID_CONCENTRATIONS.index(concentration)].predict(shear_rate_scaled)
        
        # Create a full prediction array with zeros for other concentrations
        full_predictions = np.zeros((1, 3))
        full_predictions[0, VALID_CONCENTRATIONS.index(concentration)] = prediction[0]
        
        # Inverse transform the predictions
        prediction_original = scaler_y.inverse_transform(full_predictions)
        viscosity = float(np.exp(prediction_original[0, VALID_CONCENTRATIONS.index(concentration)]))
        
        # Format response
        response = {
            'status': 'success',
            'data': {
                'shear_rate': shear_rate,
                'concentration': concentration,
                'viscosity': viscosity,
                'model': 'Support Vector Machine'
            },
            'metadata': {
                'request_id': request_id,
                'timestamp': datetime.now().isoformat(),
                'model_version': MODEL_METADATA['svm']['version']
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/ann', methods=['POST'])
def predict_ann():
    """
    Endpoint for Neural Network predictions
    Expected input format:
    {
        "shear_rate": 0.001,
        "concentration": 0.5,
        "request_id": "optional_request_id"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No input data provided',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Extract and validate input
        shear_rate = data.get('shear_rate')
        concentration = data.get('concentration')
        request_id = data.get('request_id', '')
        
        errors = validate_input(shear_rate, concentration)
        if errors:
            return jsonify({
                'status': 'error',
                'message': 'Invalid input parameters',
                'errors': errors,
                'timestamp': datetime.now().isoformat()
            }), 400

        # Preprocess input
        shear_rate_log, shear_rate_scaled = preprocess_input(shear_rate)

        # Make prediction
        prediction = ann_model.predict(shear_rate_scaled)
        # Convert prediction back to original scale
        prediction = postprocess_output(prediction)
        viscosity = float(prediction[0][VALID_CONCENTRATIONS.index(concentration)])
        
        # Format response
        response = {
            'status': 'success',
            'data': {
                'shear_rate': shear_rate,
                'concentration': concentration,
                'viscosity': viscosity,
                'model': 'Artificial Neural Network'
            },
            'metadata': {
                'request_id': request_id,
                'timestamp': datetime.now().isoformat(),
                'model_version': MODEL_METADATA['ann']['version']
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/all', methods=['POST'])
def predict_all():
    """
    Endpoint for predictions from all models
    Expected input format:
    {
        "shear_rate": 0.001,
        "concentration": 0.5,
        "request_id": "optional_request_id"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No input data provided',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Extract and validate input
        shear_rate = data.get('shear_rate')
        concentration = data.get('concentration')
        request_id = data.get('request_id', '')
        
        errors = validate_input(shear_rate, concentration)
        if errors:
            return jsonify({
                'status': 'error',
                'message': 'Invalid input parameters',
                'errors': errors,
                'timestamp': datetime.now().isoformat()
            }), 400

        # Get predictions from all models
        predictions = {}
        
        # Random Forest prediction
        shear_rate_log = np.log(shear_rate)
        X = np.array([[shear_rate_log]])
        rf_pred = rf_model.predict(X)
        rf_pred = postprocess_output(rf_pred)
        predictions['rf'] = float(rf_pred[0][VALID_CONCENTRATIONS.index(concentration)])
        
        # Gradient Boosting prediction
        gb_pred = gb_model[VALID_CONCENTRATIONS.index(concentration)].predict(X)
        gb_pred = np.exp(gb_pred)
        predictions['gb'] = float(gb_pred[0])
        
        # SVM prediction
        shear_rate_scaled = scaler_X.transform([[shear_rate_log]])
        svm_pred = svm_model[VALID_CONCENTRATIONS.index(concentration)].predict(shear_rate_scaled)
        full_predictions = np.zeros((1, 3))
        full_predictions[0, VALID_CONCENTRATIONS.index(concentration)] = svm_pred[0]
        svm_pred_original = scaler_y.inverse_transform(full_predictions)
        predictions['svm'] = float(np.exp(svm_pred_original[0, VALID_CONCENTRATIONS.index(concentration)]))
        
        # ANN prediction
        ann_pred = ann_model.predict(shear_rate_scaled)
        ann_pred = postprocess_output(ann_pred)
        predictions['ann'] = float(ann_pred[0][VALID_CONCENTRATIONS.index(concentration)])
        
        # Format response
        response = {
            'status': 'success',
            'data': {
                'shear_rate': shear_rate,
                'concentration': concentration,
                'predictions': predictions
            },
            'metadata': {
                'request_id': request_id,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """
    Endpoint to get model information and performance metrics
    """
    try:
        # Load metrics from the saved JSON file
        with open('models/model_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        response = {
            'status': 'success',
            'data': {
                'models': metrics
            }
        }
        
        return jsonify(response)
    
    except FileNotFoundError:
        return jsonify({
            'status': 'error',
            'message': 'Model metrics not found. Please train and save the models first.',
            'timestamp': datetime.now().isoformat()
        }), 404
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': all(model is not None for model in [rf_model, gb_model, svm_model, ann_model])
    })

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Check if models are loaded
    if any(model is None for model in [rf_model, gb_model, svm_model, ann_model]):
        print("Warning: Some models not loaded. Please train and save the models first.")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True) 