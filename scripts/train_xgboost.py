import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
import joblib
import os

# Create models directory if not exists
if not os.path.exists('models'):
    os.makedirs('models')

# Load Data
print("Loading data...")
try:
    df = pd.read_csv('datos/dataset_ecommerce_limpio.csv')
except FileNotFoundError:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../datos/dataset_ecommerce_limpio.csv'))

# Preprocessing
if 'CustomerID' in df.columns:
    df = df.drop('CustomerID', axis=1)

df = df.dropna()

# Identify categorical columns
# In XGBoost we can use enable_categorical=True but passing categories explicitly is safer or just use dummies
# The KNN script used dummies. To keep features aligned or at least robust, we can use dummies here too or use the native support.
# If we used Dummies in KNN, and we want to allow 'switching' models easily if they are used on the SAME feature vector in the app, 
# then we should probably assume the app sends a specific feature vector.
# However, usually apps build the vector from raw input.
# Let's use get_dummies here too for simplicity and consistency unless the notebook did otherwise.
# Notebook said "Naive Support". 
# But for dashboard simplicity, using dummies makes 'One Hot Encoding' consistent across models if we want to show side-by-side using the same preprocessor in the app.
# Actually, the app likely does its own preprocessing.
# Let's stick to get_dummies to ensure matching shapes if possible, or just treat them independently.
# The previous `app.py` implementation likely loaded these differently or handled input.
# Let's check `app.py` later. For now, creating a robust XGBoost training script.

df = pd.get_dummies(df, drop_first=True)

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save feature names order for inference
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'models/features_xgboost_churn.pkl')

# Train XGBoost
print("Training XGBoost...")
# Using scale_pos_weight for imbalance if needed, though F1 optimization was mentioned.
ratio = float(np.sum(y == 0)) / np.sum(y == 1)
model = xgb.XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=ratio,
    # Add random params reasonably close to what might be 'optimized' or just default
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"F1 Score: {f1_score(y_test, y_pred)}")

joblib.dump(model, 'models/modelo_xgboost_churn.pkl')

print("XGBoost training complete. Artifacts saved to models/")
