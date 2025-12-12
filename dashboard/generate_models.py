import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_curve, accuracy_score, roc_auc_score, f1_score

# Paths
DATA_PATH = "../datos/dataset_ecommerce_limpio.csv"
OUTPUT_DIR = "."

def generate_models():
    print("Loading dataset...")
    if not os.path.exists(DATA_PATH):
        # Fallback for local testing
        local_path = "dataset_ecommerce_limpio.csv"
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
        else:
             raise FileNotFoundError(f"Dataset not found at {DATA_PATH} or {local_path}")
    else:
        df = pd.read_csv(DATA_PATH)
    
    # ==========================================
    # 1. KNN MODEL SETUP (Existing Logic)
    # ==========================================
    print("--- Training KNN ---")
    df_knn = df.copy()
    
    # Feature Engineering for KNN
    df_knn['Es_Nuevo'] = (df_knn['Antiguedad'] < 5).astype(int)
    df_knn['Tiene_Queja'] = df_knn['Queja'].astype(int)
    df_knn['Alto_Riesgo'] = ((df_knn['Queja'] == 1) & (df_knn['Antiguedad'] < 5)).astype(int)
    df_knn['Satisfaccion_Baja'] = (df_knn['Nivel_Satisfaccion'] <= 2).astype(int)
    
    features_knn = [
        'Antiguedad', 'Dias_Ultima_Compra', 'Nivel_Satisfaccion',
        'Distancia_Almacen', 'Numero_Dispositivos', 'Monto_Cashback',
        'Queja', 'Es_Nuevo', 'Tiene_Queja', 'Alto_Riesgo', 'Satisfaccion_Baja'
    ]
    
    X_knn = df_knn[features_knn]
    y_knn = df_knn['Target'].astype(int)
    
    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
        X_knn, y_knn, test_size=0.25, random_state=42, stratify=y_knn
    )
    
    scaler = StandardScaler()
    X_train_knn_scaled = scaler.fit_transform(X_train_knn)
    X_test_knn_scaled = scaler.transform(X_test_knn)
    
    knn_model = KNeighborsClassifier(metric='manhattan', n_neighbors=5, weights='distance')
    knn_model.fit(X_train_knn_scaled, y_train_knn)
    
    # Threshold optimization for KNN
    y_proba_knn = knn_model.predict_proba(X_test_knn_scaled)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test_knn, y_proba_knn)
    df_pr = pd.DataFrame({'threshold': thresholds, 'precision': precision[:-1], 'recall': recall[:-1]})
    df_alto_recall = df_pr[df_pr['recall'] >= 0.80].copy()
    
    if not df_alto_recall.empty:
        umbral_optimo_knn = df_alto_recall.loc[df_alto_recall['precision'].idxmax()]['threshold']
    else:
        umbral_optimo_knn = 0.5
        
    print(f"KNN Optimal threshold: {umbral_optimo_knn:.4f}")
    
    # Save KNN Artifacts
    joblib.dump(knn_model, os.path.join(OUTPUT_DIR, 'modelo_knn_churn_final.pkl'))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler_knn_churn.pkl'))
    joblib.dump(features_knn, os.path.join(OUTPUT_DIR, 'features_knn_churn.pkl'))
    joblib.dump(umbral_optimo_knn, os.path.join(OUTPUT_DIR, 'umbral_optimo_knn.pkl'))
    # Save KNN test set for dashboard
    joblib.dump((X_test_knn_scaled, y_test_knn), os.path.join(OUTPUT_DIR, 'datos_test_knn.pkl'))

    # ==========================================
    # 2. XGBOOST MODEL SETUP (New Logic)
    # ==========================================
    print("--- Training XGBoost ---")
    df_xgb = df.copy()
    
    # Selected columns from notebook
    cols_xgb = ['Categoria_Preferida', 'Estado_Civil', 'Nivel_Satisfaccion',
                'Monto_Cashback', 'Queja', 'Antiguedad', 'Target']
    
    # Ensure columns exist (handling potential naming mismatches logic if needed, but assuming Clean CSV has them)
    missing_cols = [c for c in cols_xgb if c not in df_xgb.columns]
    if missing_cols:
        print(f"Warning: Missing columns for XGBoost: {missing_cols}. Skipping XGBoost.")
    else:
        df_xgb = df_xgb[cols_xgb].copy()
        
        # Convert categorical columns
        for col in ['Categoria_Preferida', 'Estado_Civil']:
            df_xgb[col] = df_xgb[col].astype('category')
            
        X_xgb = df_xgb.drop('Target', axis=1)
        y_xgb = df_xgb['Target'].astype(int)
        
        X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
            X_xgb, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb
        )
        
        # XGBoost Classifier using Hist (efficient for categorical)
        # Using simplified parameters based on notebook intent
        xgb_model = xgb.XGBClassifier(
            tree_method="hist",
            enable_categorical=True,
            random_state=42,
            n_estimators=200,    # Good default
            learning_rate=0.1,
            max_depth=5
        )
        
        xgb_model.fit(X_train_xgb, y_train_xgb)
        
        # Metrics check
        y_pred_xgb = xgb_model.predict(X_test_xgb)
        f1 = f1_score(y_test_xgb, y_pred_xgb)
        print(f"XGBoost F1-Score: {f1:.4f}")
        
        # Save XGBoost Artifacts
        joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, 'modelo_xgboost_churn.pkl'))
        # Save XGBoost test set (as dataframe for easier handling with categories in Dashboard)
        joblib.dump((X_test_xgb, y_test_xgb), os.path.join(OUTPUT_DIR, 'datos_test_xgboost.pkl'))

    print("All models generated successfully!")

if __name__ == "__main__":
    generate_models()
