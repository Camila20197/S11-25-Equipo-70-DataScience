import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from sklearn.inspection import permutation_importance
import os

# ============================
# CONFIGURACIÓN GENERAL
# ============================
st.set_page_config(page_title="Churn Dashboard – KNN Fintech", layout="wide")

st.markdown("""
<style>
.block-container {
    max-width: 900px;
    padding-top: 1rem;
}
.chart-container {
    max-width: 650px;
    margin-left: auto;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)

# ============================
# CARGA DE MODELO Y ARCHIVOS
# ============================
modelo = joblib.load("modelo_knn_churn_final.pkl")
scaler = joblib.load("scaler_knn_churn.pkl")
umbral = joblib.load("umbral_optimo_knn.pkl")
features = joblib.load("features_knn_churn.pkl")

# ============================
# CARGAR TEST SET REAL
# ============================
def cargar_test_set():
    if os.path.exists("datos_test_knn.pkl"):
        X_test_scaled, y_test = joblib.load("datos_test_knn.pkl")
        return X_test_scaled, y_test
    return None, None

X_test_scaled, y_test = cargar_test_set()

# ============================
# RECONSTRUIR EL DATASET ORIGINAL EXACTO
# ============================
df = pd.read_csv("dataset_ecommerce_limpio.csv")

# Variables derivadas EXACTAS del notebook
df["Es_Nuevo"] = (df["Antiguedad"] < 5).astype(int)
df["Tiene_Queja"] = df["Queja"].astype(int)
df["Alto_Riesgo"] = ((df["Queja"] == 1) & (df["Antiguedad"] < 5)).astype(int)
df["Satisfaccion_Baja"] = (df["Nivel_Satisfaccion"] <= 2).astype(int)

X_full = df[features]
y_full = df["Target"]

# ============================
# PERMUTATION IMPORTANCE EXACTA DEL MODELO
# ============================
X_scaled_full = scaler.transform(X_full)

result = permutation_importance(
    modelo,
    X_scaled_full,
    y_full,
    n_repeats=20,
    random_state=42
)

importancias = pd.DataFrame({
    "feature": features,
    "importance": result.importances_mean
}).sort_values("importance", ascending=True)

# ============================
# CÁLCULO DE MÉTRICAS DEL MODELO
# ============================
if X_test_scaled is not None:
    origen_metricas = "Set de prueba (igual que en el notebook)"
    proba = modelo.predict_proba(X_test_scaled)[:, 1]
    y_pred = (proba >= umbral).astype(int)
else:
    origen_metricas = "Dataset completo (NO se encontró datos_test_knn.pkl)"
    X_test_scaled = X_scaled_full
    y_test = y_full
    proba = modelo.predict_proba(X_test_scaled)[:, 1]
    y_pred = (proba >= umbral).astype(int)

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, proba)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ============================
# DASHBOARD
# ============================
st.title("Dashboard Analítico de Churn – Fintech (KNN)")

# ---- Tasa global ----
st.subheader("Métricas Principales")
tasa_churn = df["Target"].mean()
st.metric("Tasa global de churn", f"{tasa_churn:.2%}")
st.write("**Modelo cargado:**", modelo)
st.write("**Umbral óptimo:**", f"{umbral:.6f}")
st.caption(f"Métricas calculadas sobre: {origen_metricas}")

# ---- Métricas del modelo ----
st.markdown("### Desempeño del Modelo")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{acc:.2%}")
c2.metric("ROC-AUC", f"{roc:.3f}")
c3.metric("Precisión (Churn)", f"{prec:.2%}")
c4.metric("Recall (Churn)", f"{rec:.2%}")
c5.metric("F1-score", f"{f1:.2%}")

# Matriz de confusión
st.markdown("#### Matriz de Confusión")
cm_df = pd.DataFrame(
    cm,
    index=["Real 0 (No churn)", "Real 1 (Churn)"],
    columns=["Pred 0 (No churn)", "Pred 1 (Churn)"]
)
st.table(cm_df)

st.markdown("---")

# ============================
# SEGMENTACIÓN
# ============================
segmento = st.selectbox(
    "Selecciona un segmento:",
    [
        "Nivel de Satisfacción",
        "Antigüedad",
        "Distancia al Almacén",
        "Número de Dispositivos",
        "Monto Cashback",
    ]
)

df["Antiguedad_seg"] = pd.cut(df["Antiguedad"], [0, 6, 12, 18, 24, 36, 200],
                             labels=["0-6", "7-12", "13-18", "19-24", "25-36", "36+"],
                             include_lowest=True)

df["Distancia_seg"] = pd.cut(df["Distancia_Almacen"], [0, 10, 20, 30, 40, 200],
                             labels=["0-10", "11-20", "21-30", "31-40", "40+"],
                             include_lowest=True)

df["Cashback_seg"] = pd.qcut(df["Monto_Cashback"], 4,
                             labels=["Bajo", "Medio Bajo", "Medio Alto", "Alto"])

columna = {
    "Nivel de Satisfacción": "Nivel_Satisfaccion",
    "Antigüedad": "Antiguedad_seg",
    "Distancia al Almacén": "Distancia_seg",
    "Número de Dispositivos": "Numero_Dispositivos",
    "Monto Cashback": "Cashback_seg",
}[segmento]

st.subheader(f"Tasa de churn por {segmento}")

st.markdown('<div class="chart-container">', unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(5, 3))
df.groupby(columna)["Target"].mean().plot(kind="bar", color="#77c2ff", ax=ax)
ax.set_ylabel("Tasa de churn", fontsize=8)
ax.set_xlabel(columna, fontsize=8)
ax.tick_params(axis="both", labelsize=7)
plt.tight_layout(pad=0.5)
st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

# ============================
# IMPORTANCIA DE VARIABLES REAL
# ============================
st.subheader("Importancia de Variables (Permutation Importance)")

fig_imp, ax_imp = plt.subplots(figsize=(5, 3))
ax_imp.barh(importancias["feature"], importancias["importance"], color="#77c2ff")
ax_imp.set_xlabel("Impacto en el rendimiento del modelo", fontsize=8)
ax_imp.set_ylabel("Variable", fontsize=8)
ax_imp.tick_params(axis="both", labelsize=7)
plt.tight_layout(pad=0.5)
st.pyplot(fig_imp)
