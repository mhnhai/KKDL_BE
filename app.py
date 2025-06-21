from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app)

# Alternative: Enable CORS for specific origins only
# CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])

# Danh sách mô hình hợp lệ
ALLOWED_MODELS = {
    'KNN': 'models/KNN.pkl',
    'DT': 'models/DT.pkl',
    'SVM': 'models/SVM.pkl',
    'RF': 'models/RF.pkl',
    'CNB': 'models/CategoricalNB.pkl',
    'LR': 'models/LR.pkl',
    'MLP': 'models/MultinomialNB.pkl'
}

# Label mappings for validation
LABEL_MAPPINGS = {
    'parents': {0: 'Usual', 1: 'Pretentious', 2: 'Great_Pret'},
    'has_nurs': {0: 'Proper', 1: 'Improper', 2: 'Critical', 3: 'Very_Critical'},
    'form': {0: 'Completed', 1: 'Foster', 2: 'Incomplete'},
    'housing': {0: 'Convenient', 1: 'Less_Conv', 2: 'Critical'},
    'children': {0: 0, 1: 1, 2: 2, 3: 3, 4: "More"},
    'finance': {0: 'Convenient', 1: 'Inconvenient'},
    'social': {0: 'No_Problem', 1: 'Slightly_Prob', 2: 'Problematic'},
    'health': {0: 'Recommended', 1: 'Priority', 2: 'Not_Recommended'}
}

# Required features in order
REQUIRED_FEATURES = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health']


def validate_input_data(data):
    """
    Validate input data against label mappings
    Returns: (is_valid, error_message, validated_features)
    """
    errors = []
    validated_features = []

    # Check if all required features are present
    for feature in REQUIRED_FEATURES:
        if feature not in data:
            errors.append(f"Missing required feature: '{feature}'")
            continue

        value = data[feature]

        # Check if value is valid for this feature
        valid_values = list(LABEL_MAPPINGS[feature].keys())

        if value not in valid_values:
            errors.append(
                f"Invalid value for '{feature}': {value}. "
                f"Valid values are: {valid_values} "
                f"({[LABEL_MAPPINGS[feature][v] for v in valid_values]})"
            )
        else:
            validated_features.append(value)

    # Check for extra/unexpected features
    extra_features = set(data.keys()) - set(REQUIRED_FEATURES + ['model_name'])
    if extra_features:
        errors.append(f"Unexpected features found: {list(extra_features)}")

    is_valid = len(errors) == 0
    error_message = "; ".join(errors) if errors else None

    return is_valid, error_message, validated_features


@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu JSON từ frontend gửi lên
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({'error': 'Invalid JSON format'}), 400

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        # --- Validate model name ---
        model_name = data.get('model_name', '').upper()
        if not model_name:
            return jsonify({'error': 'model_name is required'}), 400

        if model_name not in ALLOWED_MODELS:
            return jsonify({
                'error': f"Model '{model_name}' not supported. Choose from: {list(ALLOWED_MODELS.keys())}"
            }), 400

        # --- Validate input features ---
        is_valid, error_message, validated_features = validate_input_data(data)

        if not is_valid:
            return jsonify({'error': f'Validation failed: {error_message}'}), 400

        # --- Load mô hình tương ứng ---
        model_path = ALLOWED_MODELS[model_name]
        if not os.path.exists(model_path):
            return jsonify({'error': f"Model file '{model_path}' not found."}), 500

        model = joblib.load(model_path)

        # Use validated features in the correct order
        input_array = np.array([validated_features])  # 2D array
        prediction = model.predict(input_array)

        return jsonify({
            'prediction': str(prediction[0]),
            'model_used': model_name,
            'input_features': dict(zip(REQUIRED_FEATURES, validated_features))
        })

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/info', methods=['GET'])
def get_info():
    return jsonify({
        'available_models': list(ALLOWED_MODELS.keys()),
        'required_features': REQUIRED_FEATURES,
        'feature_mappings': {
            feature: {
                'valid_values': list(mapping.keys()),
                'descriptions': list(mapping.values())
            }
            for feature, mapping in LABEL_MAPPINGS.items()
        }
    })


if __name__ == '__main__':
    # Use PORT environment variable (required for Cloud Run)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)