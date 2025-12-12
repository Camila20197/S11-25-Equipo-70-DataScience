import joblib
import pandas as pd
import numpy as np
import os
import xgboost

def test_dashboard_components():
    print("--- 1. Testing File Existence ---")
    required_files = [
        "modelo_knn_churn_final.pkl",
        "scaler_knn_churn.pkl",
        "features_knn_churn.pkl",
        "modelo_xgboost_churn.pkl",
        "logo.png"
    ]
    missing = []
    for f in required_files:
        if os.path.exists(f):
            print(f"[OK] Found {f}")
        else:
            print(f"[FAIL] Missing {f}")
            missing.append(f)
            
    if missing:
        print(f"CRITICAL: Missing files {missing}. Run generate_models.py first.")
        
    print("\n--- 2. Testing KNN Model Loading & Prediction ---")
    try:
        knn = joblib.load("modelo_knn_churn_final.pkl")
        scaler = joblib.load("scaler_knn_churn.pkl")
        features = joblib.load("features_knn_churn.pkl")
        
        # Mock Input
        mock_input = np.zeros((1, len(features)))
        mock_scaled = scaler.transform(mock_input)
        pred = knn.predict(mock_scaled)
        print(f"[OK] KNN Prediction successful: {pred}")
    except Exception as e:
        print(f"[FAIL] KNN Error: {e}")

    print("\n--- 3. Testing XGBoost Model Loading & Prediction ---")
    try:
        xgb = joblib.load("modelo_xgboost_churn.pkl")
        
        # Mock Input (needs valid dataframe usually, or array if pipeline)
        # XGBoost was trained on dataframe with categories. We need to match structure.
        # This part is tricky without exact columns, but let's try reading the test set pickle 
        # which we saved for exactly this reason.
        if os.path.exists("datos_test_xgboost.pkl"):
            X_test, y_test = joblib.load("datos_test_xgboost.pkl")
            pred = xgb.predict(X_test.iloc[[0]])
            print(f"[OK] XGBoost Prediction successful on test data: {pred}")
        else:
             print("[WARN] datos_test_xgboost.pkl not found, skipping prediction test")
             
    except Exception as e:
        print(f"[FAIL] XGBoost Error: {e}")

if __name__ == "__main__":
    test_dashboard_components()
