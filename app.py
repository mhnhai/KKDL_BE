from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import CategoricalNB, MultinomialNB

app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app)

# Alternative: Enable CORS for specific origins only
# CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])

# Danh sách mô hình hợp lệ
ALLOWED_MODELS = {
    'KNN': 'models/knn.pkl',
    'DT': 'models/decision_tree.pkl',
    'SVM': 'models/svm.pkl',
    'RF': 'models/random_forest.pkl',
    'CNB': 'models/categorical_naive_bayes.pkl',
    'LR': 'models/logistic.pkl',
    'MLP': 'models/multinomial_naive_bayes.pkl'
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


def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def get_model_explanation(model, model_name, input_features, prediction):
    """
    Extract meaningful information from different model types
    """
    explanation = {}

    try:
        if model_name == 'DT' and isinstance(model, DecisionTreeClassifier):
            # Decision Tree: Get tree structure and feature importances
            explanation['feature_importances'] = dict(zip(REQUIRED_FEATURES, model.feature_importances_.tolist()))
            explanation['tree_depth'] = model.tree_.max_depth
            explanation['n_leaves'] = model.tree_.n_leaves

            # Get decision path for this specific prediction
            decision_path = model.decision_path(np.array([input_features]))
            leaf_id = model.apply(np.array([input_features]))

            # Get the tree rules as text (truncated for readability)
            tree_rules = export_text(model, feature_names=REQUIRED_FEATURES, max_depth=3)
            explanation['tree_rules_sample'] = tree_rules[:500] + "..." if len(tree_rules) > 500 else tree_rules
            explanation['leaf_id'] = int(leaf_id[0])

        elif model_name == 'RF' and isinstance(model, RandomForestClassifier):
            # Random Forest: Feature importances and tree info
            explanation['feature_importances'] = dict(zip(REQUIRED_FEATURES, model.feature_importances_.tolist()))
            explanation['n_estimators'] = model.n_estimators
            explanation['max_depth'] = model.max_depth

            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(np.array([input_features]))[0]
                explanation['class_probabilities'] = {
                    str(k): float(v) for k, v in zip(model.classes_, probabilities.tolist())
                }
                explanation['confidence'] = float(max(probabilities))

        elif model_name == 'LR' and isinstance(model, LogisticRegression):
            # Logistic Regression: Coefficients and probabilities
            if hasattr(model, 'coef_'):
                # Handle both binary and multiclass cases
                if len(model.coef_) == 1:
                    explanation['feature_coefficients'] = dict(zip(REQUIRED_FEATURES, model.coef_[0].tolist()))
                else:
                    # For multiclass, show coefficients for each class
                    explanation['feature_coefficients'] = {
                        f'class_{i}': dict(zip(REQUIRED_FEATURES, coef.tolist()))
                        for i, coef in enumerate(model.coef_)
                    }
            if hasattr(model, 'intercept_'):
                explanation['intercept'] = model.intercept_.tolist() if len(model.intercept_) > 1 else float(
                    model.intercept_[0])

            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(np.array([input_features]))[0]
                explanation['class_probabilities'] = {
                    str(k): float(v) for k, v in zip(model.classes_, probabilities.tolist())
                }
                explanation['confidence'] = float(max(probabilities))

        elif model_name == 'SVM' and isinstance(model, SVC):
            # SVM: Support vectors info and decision function
            explanation['n_support_vectors'] = model.n_support_.tolist() if hasattr(model, 'n_support_') else None
            explanation['gamma'] = model.gamma if hasattr(model, 'gamma') else None
            explanation['kernel'] = model.kernel if hasattr(model, 'kernel') else None

            # Decision function (distance from hyperplane)
            if hasattr(model, 'decision_function'):
                decision_score = model.decision_function(np.array([input_features]))
                # Handle both binary and multiclass cases
                if decision_score.ndim == 1:
                    explanation['decision_score'] = float(decision_score[0])
                    explanation['confidence'] = float(abs(decision_score[0]))
                else:
                    explanation['decision_score'] = decision_score[0].tolist()
                    explanation['confidence'] = float(max(abs(decision_score[0])))

            # Probabilities if available (requires probability=True during training)
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(np.array([input_features]))[0]
                    explanation['class_probabilities'] = {
                        str(k): float(v) for k, v in zip(model.classes_, probabilities.tolist())
                    }
                except:
                    explanation['note'] = "Probabilities not available (model not trained with probability=True)"

        elif model_name == 'KNN' and isinstance(model, KNeighborsClassifier):
            # KNN: Nearest neighbors and distances
            explanation['n_neighbors'] = model.n_neighbors
            explanation['algorithm'] = model.algorithm
            explanation['metric'] = model.metric

            # Get nearest neighbors
            distances, indices = model.kneighbors(np.array([input_features]))
            explanation['neighbor_distances'] = distances[0].tolist()
            explanation['neighbor_indices'] = indices[0].tolist()

            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(np.array([input_features]))[0]
                explanation['class_probabilities'] = {
                    str(k): float(v) for k, v in zip(model.classes_, probabilities.tolist())
                }
                explanation['confidence'] = float(max(probabilities))

        elif model_name in ['CNB', 'MLP'] and isinstance(model, (CategoricalNB, MultinomialNB)):
            # Naive Bayes: Class probabilities and feature log probabilities
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(np.array([input_features]))[0]
                explanation['class_probabilities'] = {
                    str(k): float(v) for k, v in zip(model.classes_, probabilities.tolist())
                }
                explanation['confidence'] = float(max(probabilities))

            if hasattr(model, 'feature_log_prob_'):
                explanation['n_features'] = model.feature_log_prob_.shape[1]
                explanation['n_classes'] = len(model.classes_)

        elif isinstance(model, MLPClassifier):
            # MLP: Network structure and probabilities
            explanation['hidden_layer_sizes'] = model.hidden_layer_sizes
            explanation['n_layers'] = model.n_layers_
            explanation['n_outputs'] = model.n_outputs_
            explanation['activation'] = model.activation

            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(np.array([input_features]))[0]
                explanation['class_probabilities'] = {
                    str(k): float(v) for k, v in zip(model.classes_, probabilities.tolist())
                }
                explanation['confidence'] = float(max(probabilities))

        # Add general model info
        explanation['model_type'] = type(model).__name__
        explanation['input_features_used'] = dict(zip(REQUIRED_FEATURES, input_features))

    except Exception as e:
        explanation['error'] = f"Could not extract model explanation: {str(e)}"
        explanation['model_type'] = type(model).__name__

    return explanation


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

        # Get model explanation
        explanation = get_model_explanation(model, model_name, validated_features, prediction[0])

        # Convert all numpy types to native Python types before JSON serialization
        explanation = convert_numpy_types(explanation)

        return jsonify({
            'prediction': int(prediction[0]) if isinstance(prediction[0], (np.integer, np.int64)) else str(
                prediction[0]),
            'model_used': model_name,
            'input_features': {
                feature: int(value) if isinstance(value, (np.integer, np.int64)) else value
                for feature, value in zip(REQUIRED_FEATURES, validated_features)
            },
            'model_explanation': explanation
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
        },
        'model_explanations': {
            'DT': 'Returns feature importances, tree depth, decision rules, and leaf information',
            'RF': 'Returns feature importances, class probabilities, confidence scores, and ensemble info',
            'LR': 'Returns feature coefficients, intercept, class probabilities, and confidence',
            'SVM': 'Returns support vector info, decision scores, and confidence measures',
            'KNN': 'Returns neighbor distances, indices, and class probabilities',
            'CNB': 'Returns class probabilities and confidence scores',
            'MLP': 'Returns network structure info and class probabilities'
        }
    })


# Add a simple health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Backend is running'})


if __name__ == '__main__':
    # Use PORT environment variable (required for Cloud Run)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    # Uncomment the line below for production
    # app.run(host='0.0.0.0', port=port, debug=False)