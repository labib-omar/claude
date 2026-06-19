import os
import sys
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from pathlib import Path

# --- 1. INITIALIZE FLASK & MODEL ---
app = Flask(__name__, template_folder="templates", static_folder="static")

# Get the current directory where this script resides
CURRENT_DIR = Path(__file__).parent.resolve()

# CONNECTING NEW JOBLIB HERE: Points to your new LinearV4.joblib
MODEL_PATH = CURRENT_DIR.parent / "model" / "LinearV4.joblib"

print(f"Loading raw Linear Regression model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
print("✓ Model successfully initialized!")

# Retrieve the exact features the model was trained on to ensure matching column structures
try:
    # If your model stores feature names (scikit-learn 1.0+)
    MODEL_FEATURES = model.feature_names_in_
except AttributeError:
    # CRITICAL: If the above fails, paste the output list of X_train.columns from your training script here
    MODEL_FEATURES = [] 

def engineer_web_features(raw_data: dict) -> pd.DataFrame:
    """Takes web inputs and safely aligns them with the exact training feature schema

    without fragmenting the DataFrame or erasing input data.
    """
    def safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # 1. Start with a baseline row containing 0 for EVERY single expected model feature
    # This completely eliminates the loop warning and prevents resetting data to 0
    full_feature_row = {col: 0 for col in MODEL_FEATURES} if len(MODEL_FEATURES) > 0 else {}

    # 2. Extract base features from raw user web submission
    living_space = safe_float(raw_data.get('obj_livingSpace'), default=120.0)
    
    # Update numerical base columns if they exist in your model features
    if "obj_livingSpace" in full_feature_row:
        full_feature_row["obj_livingSpace"] = living_space

    # 3. Create the text categories to match against dummies
    user_regio3 = str(raw_data.get('obj_regio3', 'other'))
    user_condition = str(raw_data.get('obj_condition', 'well_kept'))

    # 4. Manually set the dummy columns to 1 for the selected categories
    # This mirrors exactly what pd.get_dummies(..., drop_first=True) did during training
    target_regio3_col = f"obj_regio3_{user_regio3}"
    target_condition_col = f"obj_condition_{user_condition}"

    if target_regio3_col in full_feature_row:
        full_feature_row[target_regio3_col] = 1
    if target_condition_col in full_feature_row:
        full_feature_row[target_condition_col] = 1

    # 5. Convert the fully prepared dictionary to a DataFrame all at once
    # If MODEL_FEATURES was empty, fallback to basic structure
    if full_feature_row:
        X_web = pd.DataFrame([full_feature_row])
        X_web = X_web[MODEL_FEATURES]  # Enforce exact training column order
    else:
        # Fallback if model features couldn't be automatically inspected
        processed_row = {
            "obj_livingSpace": living_space,
            f"obj_regio3_{user_regio3}": 1,
            f"obj_condition_{user_condition}": 1
        }
        X_web = pd.DataFrame([processed_row]).fillna(0)

    return X_web
# --- 2. ROUTES ---
@app.route('/')
def home():
    return render_template('vorhersage.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_data = request.get_json()
        df_engineered = engineer_web_features(raw_data)
        
        # Compute estimation directly using the raw Linear Regression model
        prediction = model.predict(df_engineered)[0]
        
        # Prevent unrealistic negative price predictions
        prediction = max(0, prediction)
        
        formatted_price = f"{prediction:,.2f} EUR"
        return jsonify({"predicted_price": formatted_price})
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    
    model = joblib.load("model/LinearV4.joblib")
    print(list(model.feature_names_in_))
    app.run(debug=True, port=5000)