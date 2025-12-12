import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, recall_score, make_scorer
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
    # Try absolute path or relative to script if run from root
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../datos/dataset_ecommerce_limpio.csv'))

# Preprocessing
if 'CustomerID' in df.columns:
    df = df.drop('CustomerID', axis=1)

# Handle potential missing values (though 'limpio' suggests they are handled, safe to check)
df = df.dropna()

# Encode Categorical Variables
df = pd.get_dummies(df, drop_first=True)

# Split Data
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save feature names order for inference
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'models/features_knn_churn.pkl')

# Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'models/scaler_knn_churn.pkl')

# Train KNN
# Optimizing for Recall as priority
print("Training KNN...")
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': range(3, 21, 2)}
scorer = make_scorer(recall_score)

grid = GridSearchCV(knn, param_grid, cv=5, scoring=scorer, n_jobs=-1)
grid.fit(X_train_scaled, y_train)

best_knn = grid.best_estimator_
print(f"Best params: {grid.best_params_}")

# Optimize Threshold
print("Optimizing threshold...")
y_pred_proba = best_knn.predict_proba(X_test_scaled)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.05)
best_threshold = 0.5
best_f1 = 0
target_recall = 0.8

for t in thresholds:
    y_pred_t = (y_pred_proba >= t).astype(int)
    rec = recall_score(y_test, y_pred_t)
    if rec >= target_recall:
        best_threshold = t
        # If multiple thresholds meet recall, we could pick based on precision or f1, 
        # but here we pick the highest threshold that still keeps recall high (to improve precision)
        # Actually, typically we sweep and pick. Let's stick to the one closest to optimal trade-off.
        # Simple heuristic: pick t that gives >= 0.8 recall and maximizes F1
        pass

# Refined search for best threshold maximizing F1 while Recall > 0.8
best_threshold = 0.5
max_f1_with_high_recall = 0

for t in np.arange(0.1, 0.9, 0.01):
    y_pred_t = (y_pred_proba >= t).astype(int)
    rec = recall_score(y_test, y_pred_t)
    f1 = classification_report(y_test, y_pred_t, output_dict=True)['1']['f1-score']
    
    if rec >= 0.8:
        if f1 > max_f1_with_high_recall:
            max_f1_with_high_recall = f1
            best_threshold = t

print(f"Selected Threshold: {best_threshold}")
print(f"Expected Recall: {recall_score(y_test, (y_pred_proba >= best_threshold).astype(int))}")

joblib.dump(best_knn, 'models/modelo_knn_churn_final.pkl')
joblib.dump(best_threshold, 'models/umbral_optimo_knn.pkl')

print("KNN training complete. Artifacts saved to models/")
