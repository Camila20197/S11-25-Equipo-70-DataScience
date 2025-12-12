import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from sklearn.inspection import permutation_importance

# ============================
# CONFIGURACIÓN GENERAL
# ============================
st.set_page_config(page_title="E-commerce Churn - Equipo 70", layout="wide", page_icon="logo.png")

# Estilos CSS
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        word-wrap: break-word;
        overflow-wrap: break-word;
        width: 100%;
        max-width: 100%;
    }
    .section-header {
        font-size: 1.8rem;
        border-bottom: 2px solid #2563EB;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        color: #2563EB;
    }
    .metric-container {
        background-color: #f8fafc;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stDownloadButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# FUNCIONES DE CARGA
# ============================
@st.cache_resource
def load_models():
    try:
        # KNN
        knn_model = joblib.load("modelo_knn_churn_final.pkl")
        knn_scaler = joblib.load("scaler_knn_churn.pkl")
        knn_umbral = joblib.load("umbral_optimo_knn.pkl")
        knn_features = joblib.load("features_knn_churn.pkl")
        X_test_knn_scaled, y_test_knn = joblib.load("datos_test_knn.pkl")
        
        # XGBoost
        if os.path.exists("modelo_xgboost_churn.pkl"):
            xgb_model = joblib.load("modelo_xgboost_churn.pkl")
            X_test_xgb, y_test_xgb = joblib.load("datos_test_xgboost.pkl")
        else:
            xgb_model, X_test_xgb, y_test_xgb = None, None, None

        return {
            "knn": (knn_model, knn_scaler, knn_umbral, knn_features, X_test_knn_scaled, y_test_knn),
            "xgb": (xgb_model, X_test_xgb, y_test_xgb)
        }
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None

@st.cache_data
def load_data():
    try:
        path = "../datos/dataset_ecommerce_limpio.csv"
        if not os.path.exists(path):
            path = "dataset_ecommerce_limpio.csv"
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()

# Cargar
models_data = load_models()
df = load_data()

# ============================
# SIDEBAR
# ============================
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.title("ChurnZero")
    
    st.markdown("---")
    
    menu = st.radio( 
        "Navegación",
        [
            "1. Análisis exploratorio de datos (EDA)",
            "2. Definición del churn",
            "3. Modelado Predictivo",
            "4. Segmentación de clientes",
            "5. Dashboard analítico",
            "6. Recomendaciones de acción (insights de negocio)",
            "7. Predicción de Fuga"
        ]
    )
    
    st.markdown("---")
    st.info("**S11-25-Equipo-70-*\nDataScience")

# ============================
# LÓGICA DE SECCIONES
# ============================

# -----------------
# -----------------
# 1. EDA
# -----------------
if "1. Análisis exploratorio" in menu:
    st.markdown('<div class="main-header">Análisis Exploratorio de Datos (EDA)</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Resumen General", "Comportamiento", "Correlaciones"])
    
    with tab1:
        st.markdown("### Proceso de Limpieza y Preparación de Datos")
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros Totales", f"{df.shape[0]:,}")
        c2.metric("Variables", df.shape[1])
        c3.metric("Tasa de Churn Base", f"{df['Target'].mean():.1%}")
        
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("""
            #### Transformaciones Aplicadas
            
            **1. Renombr amiento de Variables**
            - Tenure → Antiguedad
            - WarehouseToHome → Distancia_Almacen
            - NumberOfDeviceRegistered → Numero_Dispositivos
            - PreferedOrderCat → Categoria_Preferida
            - SatisfactionScore → Nivel_Satisfaccion
            - MaritalStatus → Estado_Civil
            - DaySinceLastOrder → Dias_Ultima_Compra
            - CashbackAmount → Monto_Cashback
            - Churn → Target
            
            **2. Conversión de Tipos**
            - Variables categóricas → string (minúsculas)
            - Variables binarias → int (0/1)
            """)
        
        with col_b:
            st.markdown("""
            #### Validación de Calidad
            
            **Valores Nulos (Imputados con Mediana)**:
            - `Antiguedad`: ✓ Imputado
            - `Distancia_Almacen`: ✓ Imputado
            - `Dias_Ultima_Compra`: ✓ Imputado
            
            **Duplicados**: ✓ No se encontraron
            
            **Prueba de Normalidad (Kolmogorov-Smirnov)**:
            - Todas las variables: p-valor < 0.05
            - **Conclusión**: Ninguna variable sigue distribución normal
            - **Implicación**: Preferir modelos no paramétricos
            """)
        
        st.markdown("---")
        st.markdown("### Detección de Outliers")
        
        col_out1, col_out2 = st.columns(2)
        with col_out1:
            outlier_data = pd.DataFrame({
                "Variable": ["Numero_Dispositivos", "Monto_Cashback", "Distancia_Almacen", "Dias_Ultima_Compra"],
                "% Outliers": ["6.8%", "8.0%", "0.35%", "0.5%"]
            })
            st.dataframe(outlier_data, use_container_width=True, hide_index=True)
            
        with col_out2:
            st.info("""
            **Análisis Multivariado (Isolation Forest)**
            
            - Outliers detectados: **15 registros**
            - Basado en Distancia de Mahalanobis
            - Estos registros presentan patrones atípicos en múltiples dimensiones
            """)
    
    with tab2:
        st.markdown("### Patrones de Comportamiento")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Churn por Categoría Preferida**")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.countplot(y="Categoria_Preferida", hue="Target", data=df, palette="viridis", ax=ax)
            ax.set_title("Abandono según Categoría")
            st.pyplot(fig)
            
        with col2:
            st.markdown("**Distribución de Días desde Última Compra**")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(data=df, x="Dias_Ultima_Compra", hue="Target", kde=True, palette="coolwarm", ax=ax)
            ax.set_title("Recencia de Compra vs Churn")
            st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("### Diagramas de Cajas (Outliers y Distribuciones)")
        col3, col4 = st.columns(2)
        with col3:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.boxplot(data=df, y="Monto_Cashback", x="Target", palette="Set2", ax=ax)
            ax.set_title("Cashback por Estado de Churn")
            ax.set_xlabel("Churn (0=No, 1=Sí)")
            st.pyplot(fig)
        
        with col4:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.boxplot(data=df, y="Antiguedad", x="Target", palette="Set3", ax=ax)
            ax.set_title("Antigüedad por Estado de Churn")
            ax.set_xlabel("Churn (0=No, 1=Sí)")
            st.pyplot(fig)

    with tab3:
        st.markdown("### Análisis de Correlaciones y Reducción Dimensional")
        
        col_corr1, col_corr2 = st.columns([2, 1])
        
        with col_corr1:
            st.markdown("#### Mapa de Calor de Correlaciones")
            if not df.empty:
                # Select numeric columns only
                numeric_df = df.select_dtypes(include=[np.number])
                corr = numeric_df.corr()
                fig,ax = plt.subplots(figsize=(8,6))
                sns.heatmap(corr, cmap="RdBu_r", center=0, linewidths=0.5, fmt=".1f", ax=ax)
                st.pyplot(fig)
        
        with col_corr2:
            st.markdown("#### Variables Clave vs Churn")
            st.success("""
            **Correlación Positiva** (↑ Riesgo)
            - **Queja**: +0.25
            - **Dias_Ultima_Compra**: +0.09
            """)
            
            st.error("""
            **Correlación Negativa** (↓ Riesgo)
            - **Antiguedad**: -0.35 ⭐
            - **Monto_Cashback**: -0.16
            - **Nivel_Satisfaccion**: -0.03
            """)
            
            st.info("""
            **Insight Principal**:
            Los clientes nuevos con quejas tienen el mayor riesgo de churn.
            """)
        
        st.markdown("---")
        st.markdown("#### Análisis de Componentes Principales (PCA)")
        
        col_pca1, col_pca2 = st.columns(2)
        with col_pca1:
            st.markdown("""
            **Objetivo**: Evaluar separabilidad lineal de las clases
            
            **Configuración**:
            - Reducción a 2 componentes principales
            - Variables numéricas estandarizadas
            
            **Resultados**:
            - Varianza explicada: ~50%
            - Las clases Churn/No Churn están **mezcladas**
            """)
        
        with col_pca2:
            st.warning("""
            **Conclusión del PCA**:
            
            El problema NO es linealmente separable en bajas dimensiones.
            
            **Implicación**:
            - Descartar modelos lineales simples
            - Usar modelos no lineales (KNN, XGBoost, RF)
            - Requiere ingeniería de features avanzada
            """)

# -----------------
# 2. DEFINICIÓN CHURN
# -----------------
elif "2. Definición" in menu:
    st.markdown('<div class="main-header">Definición del Churn</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Objetivo del Negocio
    Desarrollar un modelo para identificar patrones de abandono y detectar señales tempranas de fuga, permitiendo implementar estrategias de retención proactivas y personalizadas.
    """)
    
    st.markdown("---")
    
    # Hallazgo principal
    st.error("""
    ### Hallazgo Crítico
    
    **El Churn NO es por inactividad**: La media de días desde la última compra es muy baja (~4.5 días). 
    
    El abandono suele ocurrir **poco después de una compra**, sugiriendo **insatisfacción reciente** con el producto/servicio.
    """)
    
    st.markdown("---")
    
    # Factores de riesgo
    st.markdown("### Factores de Riesgo Cuantificados")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.metric("Impacto de Quejas", "3x", help="Las quejas triplican la probabilidad de churn")
    with col_f2:
        st.metric("Riesgo Clientes Nuevos", "5.5x", help="Clientes con < 5 meses tienen 5.5x más riesgo")
    with col_f3:
        st.metric("Caso Crítico", "> 60%", help="Nuevos con quejas: tasa de churn > 60%")
    
    st.markdown("---")
    
    # Tres definiciones operativas
    st.markdown("### Definiciones Operativas de Churn")
    
    tab_a, tab_b, tab_c = st.tabs(["Definición A: Churn Explícito", "Definición B: Alto Riesgo", "Definición C: Inactividad Atípica"])
    
    with tab_a:
        st.markdown("""
        #### Target (Variable Objetivo)
        
        **Criterio**: `Target == 1`
        
        **Proporción**: 17.1% de la base de clientes
        
        **Uso**: 
        - Entrenamiento de modelos predictivos
        - Variable objetivo para algoritmos de ML
        - Evaluación de métricas de rendimiento
        
        **Características**:
        - Clientes que ya abandonaron el servicio
        - Basado en comportamiento histórico confirmado
        - Permite aprendizaje supervisado
        """)
    
    with tab_b:
        st.warning("""
        #### Alerta Temprana - Intervención Inmediata
        
        **Criterio**: `(Queja == 1) & (Antiguedad < 5 meses)`
        
        **Proporción**: 10.4% de la base de clientes
        
        **Acción Recomendada**: 
        - ⚡ Intervención inmediata por Customer Success
        - Contacto personal dentro de 24-48 horas
        - Descuento de compensación 10-15%
        - Escalamiento a gerencia si es necesario
        
        **Justificación**:
        - Combinación de factores con mayor tasa de churn
        - Ventana crítica de retención
        - Alto ROI de intervención temprana
        """)
    
    with tab_c:
        st.info("""
        #### Reactivación - Inactividad Inusual
        
        **Criterio**: `Dias_Ultima_Compra > 15 días`
        
        **Proporción**: 0.8% de la base de clientes
        
        **Acción Recomendada**:
        - 📧 Campaña de reactivación por email
        - Oferta especial de "Te extrañamos"
        - Recordatorio de beneficios/cashback acumulado
        
        **Contexto**:
        - 99% de clientes activos compran antes de 15 días
        - Patrón atípico indica posible abandono silencioso
        - Bajo volumen permite atención personalizada
        """)
    
    st.markdown("---")
    
    # Composición del dataset
    st.markdown("### Composición del Dataset")
    col_data1, col_data2 = st.columns(2)
    
    with col_data1:
        st.markdown("""
        **Variables Demográficas**:
        - Estado Civil
        - Ubicación (Distancia al Almacén)
        
        **Variables de Comportamiento**:
        - Cashback acumulado
        - Categorías preferidas
        - Número de dispositivos registrados
        - Antigüedad como cliente
        """)
    
    with col_data2:
        st.markdown("""
        **Variables de Satisfacción**:
        - Nivel de Satisfacción (1-5)
        - Quejas registradas (Sí/No)
        - Días desde última compra
        
        **Total Registros**: 3,941 clientes
        **Periodo**: Datos históricos de e-commerce
        **Tasa Base de Churn**: 17.1%
        """)

# -----------------
# 3. MODELADO PREDICTIVO
# -----------------
elif "3. Modelado Predictivo" in menu:
    st.markdown('<div class="main-header">Modelado Predictivo</div>', unsafe_allow_html=True)
    
    # Metodología
    st.markdown("### Metodología y Optimización")
    
    col_meth1, col_meth2 = st.columns(2)
    
    with col_meth1:
        st.markdown("""
        #### Estrategia de Modelado
        
        **Configuración General**:
        - **División de datos**: Train 75% / Test 25%
        - **Estratificación**: Mantener balance de clases
        - **Métrica principal**: F1-Score (balance precisión/recall)
        
        **Modelos Evaluados**:
        1. **K-Nearest Neighbors (KNN)**
           - Feature Engineering intensivo
           - Optimización de umbral
           - Escalado StandardScaler
           - Distancia: Manhattan
        
        2. ** XGBoost Classifier**
           - RandomizedSearchCV (100 iteraciones)
           - scale_pos_weight para desbalance
           - Tree method: hist
           - Features categóricas nativas
        """)
    
    with col_meth2:
        st.markdown("""
        #### Feature Engineering (KNN)
        
        Se crearon **4 variables derivadas** para mejorar el rendimiento:
        
        - `Es_Nuevo`: Cliente con < 5 meses antigüedad
        - `Tiene_Queja`: Binario de queja
        - `Alto_Riesgo`: Nuevo + Queja (combinación crítica)
        - `Satisfaccion_Baja`: Nivel ≤ 2
        
        **Variables Finales KNN** (11 total):
        - Antiguedad, Dias_Ultima_Compra
        - Nivel_Satisfaccion, Distancia_Almacen
        - Numero_Dispositivos, Monto_Cashback
        - Queja + 4 features derivados
        
        #### Hiperparámetros XGBoost
        - `n_estimators`: 463
        - `max_depth`: 5
        - `learning_rate`: 0.205
        - `subsample`: 0.81
        - `colsample_bytree`: 0.76
        """)
    
    st.markdown("---")
    
    if models_data is None:
        st.error("No se pudieron cargar los modelos.")
        st.stop()
        
    knn_data = models_data["knn"]
    xgb_data = models_data["xgb"]
    
    st.markdown("### Evaluación de Modelos")
    algo = st.selectbox("Seleccionar Algoritmo", ["k-Nearest Neighbors (KNN)", "XGBoost (Gradient Boosting)"])
    
    col_metrics, col_viz = st.columns([1, 2])
    
    if algo == "k-Nearest Neighbors (KNN)":
        modelo, scaler, umbral, features, X_test, y_test = knn_data
        
        # Predicciones
        proba = modelo.predict_proba(X_test)[:, 1]
        y_pred = (proba >= umbral).astype(int)
        
        with col_metrics:
            st.info("Algo: KNN\nEscalado: StandardScaler\nDistancia: Manhattan")
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
            st.metric("F1-Score", f"{f1_score(y_test, y_pred):.1%}")
            st.metric("ROC-AUC", f"{roc_auc_score(y_test, proba):.3f}")
            
    else: # XGBoost
        if xgb_data[0] is None:
            st.warning("Modelo XGBoost no encontrado. Ejecuta `generate_models.py`.")
            st.stop()
            
        modelo, X_test, y_test = xgb_data
        y_pred = modelo.predict(X_test)
        proba = modelo.predict_proba(X_test)[:, 1] # Puede variar según versión xgb
        
        with col_metrics:
            st.info("Algo: XGBoost\nBoosters: Hist\nCategorical Support: On")
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
            st.metric("F1-Score", f"{f1_score(y_test, y_pred):.1%}")
            st.metric("ROC-AUC", f"{roc_auc_score(y_test, proba):.3f}")

    with col_viz:
        st.markdown("#### Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Retenido','Churn'], yticklabels=['Real 0','Real 1'])
        st.pyplot(fig)
        
    st.markdown("### Exportabilidad de Resultados")
    st.markdown("Descarga las predicciones del set de prueba para análisis externo.")
    
    # Crear DF para descarga
    df_export = pd.DataFrame({
        "Real_Target": y_test,
        "Prediccion": y_pred,
        "Probabilidad_Fuga": proba
    })
    
    csv = df_export.to_csv(index=False).encode('utf-8')
    st.download_button(
        " Descargar Predicciones (CSV)",
        csv,
        "predicciones_churn.csv",
        "text/csv",
        key='download-csv'
    )
    
    # Comparación de Modelos
    st.markdown("---")
    st.markdown("### Comparación de Modelos y Selección Final")
    
    if models_data is not None:
        st.info("""**Modelo Seleccionado: K-Nearest Neighbors (KNN)**
        
Basándonos en el análisis comparativo de ambos modelos, seleccionamos KNN como modelo final por las siguientes razones:
        
1. **Mayor Recall (Sensibilidad)**: KNN logra identificar un mayor porcentaje de clientes que realmente harán churn (~87-88%), crucial para acciones preventivas.
2. **F1-Score Superior**: Con un F1-Score del 87.5%, KNN mantiene un mejor equilibrio entre precisión y recall.
3. **Interpretabilidad**: El enfoque basado en vecinos cercanos es más intuitivo para el equipo de negocio.
        
El modelo XGBoost, aunque tiene mayor precisión general (89.3%), detecta menos casos de churn real, lo cual es crítico en estrategias de retención.""")
        
        # Tabla comparativa
        knn_data = models_data["knn"]
        xgb_data = models_data["xgb"]
        
        if xgb_data[0] is not None:
            modelo_knn, scaler, umbral, features, X_test_knn, y_test_knn = knn_data
            modelo_xgb, X_test_xgb, y_test_xgb = xgb_data
            
            proba_knn = modelo_knn.predict_proba(X_test_knn)[:, 1]
            y_pred_knn = (proba_knn >= umbral).astype(int)
            
            y_pred_xgb = modelo_xgb.predict(X_test_xgb)
            proba_xgb = modelo_xgb.predict_proba(X_test_xgb)[:, 1]
            
            comparison_df = pd.DataFrame({
                "Métrica": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                "KNN ⭐": [
                    f"{accuracy_score(y_test_knn, y_pred_knn):.1%}",
                    f"{precision_score(y_test_knn, y_pred_knn):.1%}",
                    f"{recall_score(y_test_knn, y_pred_knn):.1%}",
                    f"{f1_score(y_test_knn, y_pred_knn):.1%}",
                    f"{roc_auc_score(y_test_knn, proba_knn):.3f}"
                ],
                "XGBoost": [
                    f"{accuracy_score(y_test_xgb, y_pred_xgb):.1%}",
                    f"{precision_score(y_test_xgb, y_pred_xgb):.1%}",
                    f"{recall_score(y_test_xgb, y_pred_xgb):.1%}",
                    f"{f1_score(y_test_xgb, y_pred_xgb):.1%}",
                    f"{roc_auc_score(y_test_xgb, proba_xgb):.3f}"
                ]
            })
            
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# -----------------
# 4. SEGMENTACIÓN
# -----------------
elif "4. Segmentación" in menu:
    st.markdown('<div class="main-header">Segmentación de Clientes</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        **Niveles de Riesgo:**
        *   🔴 **Alto**: Prob > 70%
        *   🟡 **Medio**: Prob 30-70%
        *   🟢 **Bajo**: Prob < 30%
        """)
        segment_var = st.selectbox("Segmentar por:", ["Nivel_Satisfaccion", "Queja", "Categoria_Preferida"])
        
    with col2:
        # Simple aggregation viz
        df_ag = df.groupby(segment_var)["Target"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(x=segment_var, y="Target", data=df_ag, palette="OrRd", ax=ax)
        ax.set_ylabel("Tasa de Churn")
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0%')
        st.pyplot(fig)

# -----------------
# 5. DASHBOARD ANALÍTICO
# -----------------
elif "5. Dashboard analítico" in menu:
    st.markdown('<div class="main-header">Dashboard Analítico</div>', unsafe_allow_html=True)
    
    st.markdown("### Visión General del Negocio")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clientes Activos", f"{len(df[df['Target']==0]):,}", delta=f"-{len(df[df['Target']==1])} fugados")
    k2.metric("Tasa de Fuga Actual", f"{df['Target'].mean():.1%}", help="Porcentaje de clientes que han abandonado")
    k3.metric("Ingresos en Riesgo (Est.)", "$45,000", delta="-12%", delta_color="inverse", help="Estimación basada en cashback promedio x clientes en fuga")
    k4.metric("Satisfacción Promedio", f"{df['Nivel_Satisfaccion'].mean():.1f}/5.0")
    
    st.markdown("---")
    
    st.markdown("### Inteligencia del Modelo Predictivo")
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.info("**Capacidad de Detección (Recall)**")
        st.write("## 87.5%")
        st.caption("De cada 10 clientes que se van, el modelo detecta 9 a tiempo.")
        
    with m2:
        st.success("**Precisión General**")
        st.write("## 89.3%")
        st.caption("Nivel de confiabilidad global de las predicciones.")
        
    with m3:
        st.warning("**Ahorro Potencial**")
        st.write("## 65%")
        st.caption("Del churn total, podemos evitar el 65% con acciones preventivas.")

    st.markdown("---")
    st.markdown("###  Factores Críticos Visualizados")

    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.markdown("**Impacto de las Quejas en la Fuga**")
        # Gráfico simple de Quejas vs Churn
        churn_quejas = df.groupby('Queja')['Target'].mean()
        fig_q, ax_q = plt.subplots(figsize=(6,4))
        bars = ax_q.bar(["Sin Quejas", "Con Quejas"], churn_quejas.values, color=['#4ADE80', '#F87171'])
        ax_q.set_ylabel("Probabilidad de Fuga")
        ax_q.bar_label(bars, fmt='%.0%')
        st.pyplot(fig_q)
        st.caption("💡 Los clientes con quejas tienen el TRIPLE de riesgo de irse.")

    with col_viz2:
        st.markdown("**Ciclo de Vida: ¿Cuándo se van más?**")
        # Binning antiguedad mejorado
        df['Ciclo_Vida'] = pd.cut(df['Antiguedad'], 
                                   bins=[-1, 1, 6, 12, 24, 60], 
                                   labels=['Mes 1', '1-6 Meses', '6-12 Meses', '1-2 Años', '+2 Años'])
        churn_ciclo = df.groupby('Ciclo_Vida')['Target'].mean()
        
        fig_c, ax_c = plt.subplots(figsize=(6,4))
        sns.lineplot(x=churn_ciclo.index, y=churn_ciclo.values, marker='o', linewidth=3, color='#3B82F6', ax=ax_c)
        ax_c.set_ylabel("Tasa de Fuga")
        ax_c.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_c)
        st.caption("💡 El riesgo es crítico en los primeros 6 meses (Onboarding).")
        
    # Segunda fila de gráficos
    col_viz3, col_viz4 = st.columns(2)
    
    with col_viz3:
        st.markdown("**Relación Satisfacción vs Fuga**")
        # Agrupar por satisfaccion
        churn_sat = df.groupby('Nivel_Satisfaccion')['Target'].mean() * 100
        
        fig_s, ax_s = plt.subplots(figsize=(6,4))
        # Colores semaforo: Rojo a Verde
        colors_sat = ['#EF4444', '#F97316', '#EAB308', '#84CC16', '#22C55E']
        sns.barplot(x=churn_sat.index, y=churn_sat.values, palette=colors_sat, ax=ax_s)
        ax_s.set_ylabel("% de Fuga")
        ax_s.set_xlabel("Nivel de Satisfacción (1-5)")
        ax_s.set_ylim(0, 30) # Fix scale for comparison
        for container in ax_s.containers:
            ax_s.bar_label(container, fmt='%.1f%%')
        st.pyplot(fig_s)
        st.caption("💡 Clientes insuficientes (1-2 estrellas) se van un **50% más**.")

    with col_viz4:
        st.markdown("**¿El dinero compra lealtad? (Cashback)**")
        # Binning cashback
        df['Cashback_Bin'] = pd.cut(df['Monto_Cashback'], bins=[0, 150, 200, 300, 1000], labels=['Bajo (<$150)', 'Medio', 'Alto', 'Premium (>$300)'])
        churn_cash = df.groupby('Cashback_Bin')['Target'].mean() * 100
        
        fig_ca, ax_ca = plt.subplots(figsize=(6,4))
        sns.lineplot(x=churn_cash.index, y=churn_cash.values, marker='o', color='#8B5CF6', linewidth=3, ax=ax_ca)
        # Fill area under line
        ax_ca.fill_between(churn_cash.index, churn_cash.values, color='#8B5CF6', alpha=0.1)
        ax_ca.set_ylabel("% de Fuga")
        st.pyplot(fig_ca)
        st.caption("💡 **Sí**. A mayor cashback acumulado, drásticamente MENOR fuga.")
        
    st.markdown("---")
    st.markdown("### ⚙️ ¿Cómo funciona este análisis? (El Proceso)")
    
    st.info("""
    Este dashboard no es magia, es **Ciencia de Datos aplicada**. Así procesamos la información de tus clientes:
    """)
    
    c_p1, c_p2, c_p3, c_p4 = st.columns(4)
    
    with c_p1:
        st.markdown("#### 1. Ingesta")
        st.markdown("")
        st.caption("Recopilamos datos históricos de compras, quejas y comportamiento web.")
        
    with c_p2:
        st.markdown("#### 2. Limpieza")
        st.markdown("")
        st.caption("Corregimos errores, completamos datos faltantes y filtramos 'ruido'.")
        
    with c_p3:
        st.markdown("#### 3. Aprendizaje")
        st.markdown("")
        st.caption("El modelo (KNN) estudia patrones de 4,000 clientes para aprender qué causa la fuga.")
        
    with c_p4:
        st.markdown("#### 4. Predicción")
        st.markdown("")
        st.caption("Aplicamos lo aprendido para alertar sobre riesgos en tiempo real.")

# -----------------
# 6. RECOMENDACIONES
# -----------------
elif "6. Recomendaciones" in menu:
    st.markdown('<div class="main-header">Recomendaciones de Acción (Insights de Negocio)</div>', unsafe_allow_html=True)
    
    st.markdown("### Estrategias Priorizadas por Impacto")
    
    # Overview de estrategias
    col_strat1, col_strat2, col_strat3 = st.columns(3)
    
    with col_strat1:
        st.metric("Quick Wins", "2 estrategias", delta="Implementación < 1 mes", help="Alerta Quejas y Reactivación Inactividad")
    with col_strat2:
        st.metric("Mediano Plazo", "2 estrategias", delta="2-3 meses", help="Onboarding y Cashback Tier")
    with col_strat3:
        st.metric("ROI Estimado", "3.5x - 5x", delta="Retención vs Adquisición", help="Retener cuesta 5x menos que adquirir")
    
    st.markdown("---")
    
    tab_quick, tab_medium, tab_long = st.tabs([" Quick Wins (< 1 mes)", " Mediano Plazo (2-3 meses)", " Largo Plazo (6+ meses)"])
    
    with tab_quick:
        st.markdown("### Estrategia 1: Programa 'Alerta Quejas'")
        
        col_q1, col_q2 = st.columns(2)
        
        with col_q1:
            st.markdown("""
            #### Análisis de Datos
            
            **Hallazgo**:
            - Clientes con quejas: **3x más riesgo** de churn
            - Combinación nuevos + quejas: **> 60% churn rate**
            - Afecta al 10.4% de la base
            
            **Segmento Objetivo**:
            - Clientes con < 6 meses antigüedad
            - Queja registrada (cualquier categoría)
            - Estado: Activo pero en riesgo crítico
            """)
        
        with col_q2:
            st.markdown("""
            #### Plan de Acción
            
            **Implementación**:
            1. Sistema de alertas automáticas
            2. SLA 24h para contacto inicial
            3. Script de recuperación preparado
            
            **Incentivo**:
            - Descuento compensatorio 10-15%
            - Cashback extra en próxima compra
            - Upgrade temporal a tier premium
            
            **ROI Estimado**: 4.5x
            **Costo por cliente**: $15-20
            **Valor vida útil recuperado**: $70-90
            """)
        
        st.markdown("---")
        st.markdown("### Estrategia 2: Campaña Reactivación Inactividad")
        
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.info("""
            **Segmento**: Clientes sin compra en 15+ días (0.8% base)
            
            **Características**:
            - Patrón atípico (99% compra antes de 15 días)
            - Bajo volumen permite personalización
            - Alta probabilidad de abandono silencioso
            """)
        
        with col_r2:
            st.success("""
            **Acciones**:
            1. Email personalizado "Te extrañamos"
            2. Cupón 20% válido 72 horas
            3. Reminder de cashback acumulado
            4. Recomendaciones basadas en historial
            
            **Automatización**: Sí (trigger por días)
            **Frecuencia**: Cada 48 horas hasta compra
            """)
    
    with tab_medium:
        st.markdown("### Estrategia 3: Onboarding Intensivo (Primeros 90 días)")
        
        col_o1, col_o2 = st.columns(2)
        
        with col_o1:
            st.markdown("""
            ####  Objetivo
            Reducir churn de clientes nuevos (< 6 meses) del 35% al 20% en 3 meses.
            
            #### 📅 Customer Journey
            
            **Día 0**: Bienvenida + Tutorial
            - Video guía de la plataforma
            - Beneficios cashback explicados
            - Primer cupón 15% OFF
            
            **Día 7**: Educación
            - Tips de compra inteligente
            - Categorías más populares
            - Caso de estudio de ahorro
            
            **Día 14**: Engagement
            - Notificación de productos relacionados
            - Invitación a dejar review
            - Bonus por segunda compra
            
            **Día 30**: Fidelización
            - Email con estadísticas personales
            - Cashback acumulado destacado
            - Unlock de beneficio exclusivo
            """)
        
        with col_o2:
            st.markdown("""
            #### 💰 Inversión & ROI
            
            **Costos**:
            - Ingreso adicional: $108,000/año
            
            **ROI**: **12.7x** en primer año
            
            #### Métricas de Seguimiento
            - Tasa de apertura emails (objetivo: > 35%)
            - CTR en CTAs (objetivo: > 8%)
            - Tasa segunda compra día 30 (objetivo: > 55%)
            - NPS nuevos clientes (objetivo: > 7)
            """)
    
    with tab_long:
        st.markdown("### Estrategia 4: Programa de Cashback por Tiers")
        st.info("""
        **Concepto**: Sistema de niveles basado en compras acumuladas.
        
        **Tiers**:
        - Bronze (0-$500): 2% cashback
        - Silver ($500-$1500): 3.5% cashback + envío gratis
        - Gold ($1500+): 5% cashback + soporte prioritario + acc eso anticipado ofertas
        
        **Impacto Esperado**: Incremento retención 18-22% en clientes Silver/Gold
        **Timeline**: 6 meses desarrollo + piloto + rollout
        **Inversión**: $25,000-35,000
        """)

# -----------------
# 7. PREDICCIÓN (SIMULADOR)
# -----------------
elif "7. Predicción" in menu:
    st.markdown('<div class="main-header">Predicción de Fuga en Tiempo Real</div>', unsafe_allow_html=True)
    st.markdown("### Predice si un cliente abandonará la empresa")
    st.info("Ingresa los datos del cliente para obtener una predicción instantánea basada en nuestro modelo de Inteligencia Artificial (KNN).")

    col_input1, col_input2 = st.columns(2)

    with col_input1:
        antiguedad = st.number_input("Antigüedad (meses)", min_value=0, max_value=120, value=1, help="Meses que el cliente lleva registrado en la plataforma.")
        dias_compra = st.number_input("Días desde última compra", min_value=0, max_value=365, value=5, help="Días pasados desde que el cliente hizo su último pedido.")
        satisfaccion = st.slider("Nivel de Satisfacción (1-5)", 1, 5, 3, help="Puntaje de satisfacción del cliente en encuestas.")
        distancia = st.number_input("Distancia al Almacén (km)", min_value=0.0, max_value=100.0, value=10.0, help="Distancia entre el domicilio del cliente y el almacén.")

    with col_input2:
        cashback = st.number_input("Monto de Cashback Acumulado ($)", min_value=0.0, max_value=1000.0, value=150.0, help="Dinero devuelto acumulado por compras anteriores.")
        dispositivos = st.number_input("Dispositivos Registrados", min_value=1, max_value=10, value=2, help="Número de dispositivos vinculados a la cuenta.")
        queja = st.selectbox("¿Ha realizado alguna queja reciente?", ["No", "Sí"], help="Indica si el cliente ha levantado un ticket de queja recientemente.")
        queja_val = 1 if queja == "Sí" else 0

    st.markdown("---")

    if st.button(" Calcular Probabilidad de Fuga", type="primary"):
        # Cargar recursos KNN
        knn_data = models_data["knn"]
        if knn_data:
            knn_model, knn_scaler, knn_umbral, knn_features, _, _ = knn_data

            # Crear dataframe input
            input_dict = {
                'Antiguedad': [antiguedad],
                'Dias_Ultima_Compra': [dias_compra],
                'Nivel_Satisfaccion': [satisfaccion],
                'Distancia_Almacen': [distancia],
                'Numero_Dispositivos': [dispositivos],
                'Monto_Cashback': [cashback],
                'Queja': [queja_val]
            }
            
            # Feature Engineering On-the-fly (REPLICANDO LOGICA DE ENTRENAMIENTO)
            input_df = pd.DataFrame(input_dict)
            input_df['Es_Nuevo'] = (input_df['Antiguedad'] < 5).astype(int)
            input_df['Tiene_Queja'] = input_df['Queja'].astype(int)
            input_df['Alto_Riesgo'] = ((input_df['Queja'] == 1) & (input_df['Antiguedad'] < 5)).astype(int)
            input_df['Satisfaccion_Baja'] = (input_df['Nivel_Satisfaccion'] <= 2).astype(int)
            
            # Ordenar columnas según entrenamiento
            try:
                input_final = input_df[knn_features]
                
                # Escalar
                input_scaled = knn_scaler.transform(input_final)
                
                # Predecir
                probabilidad = knn_model.predict_proba(input_scaled)[0, 1]
                prediccion = 1 if probabilidad >= knn_umbral else 0
                
                st.markdown("#### Resultado del Análisis")
                
                c_res1, c_res2 = st.columns([1, 2])
                
                with c_res1:
                    if prediccion == 1:
                        st.error(f"🚨 ALTO RIESGO DE FUGA")
                        st.metric("Probabilidad de Abandono", f"{probabilidad:.1%}")
                    else:
                        st.success(f"✅ CLIENTE SEGURO")
                        st.metric("Probabilidad de Abandono", f"{probabilidad:.1%}")
                
                with c_res2:
                    st.markdown("** Análisis y Recomendaciones Personalizadas:**")
                    
                    if prediccion == 1:
                        st.markdown(" **Diagnóstico: RIESGO DE FUGA DETECTADO**")
                        st.markdown("Se recomienda activar los siguientes protocolos:")
                        
                        # Recomendaciones específicas basadas en variables
                        recos = []
                        
                        if queja_val == 1:
                             recos.append("- **Gestión de Quejas**: El cliente tiene un reclamo activo. Asignar ticket prioritario a Soporte y realizar seguimiento telefónico en 24hs.")
                        
                        if satisfaccion <= 2:
                             recos.append("- **Mejora de Experiencia**: La satisfacción es crítica (≤2). Enviar encuesta de profundidad u ofrecer cupón de disculpas del 15%.")
                        
                        if antiguedad < 5:
                             recos.append("- **Onboarding**: Cliente nuevo en riesgo. Reforzar comunicación de valor y tutoriales de uso.")
                        elif antiguedad > 24:
                             recos.append("- **Fidelización**: Cliente antiguo en riesgo. Verificar si hay agotamiento o mejores ofertas de la competencia. Ofrecer 'Upgrade' de servicio.")
                             
                        if dias_compra > 15:
                             recos.append("- **Reactivación**: Inactividad reciente. Enviar email 'Te extrañamos' con productos complementarios a su última compra.")
                             
                        if cashback < 100:
                             recos.append("- **Incentivos**: Saldo de cashback bajo. Ofrecer bono de $50 por próxima compra para aumentar el coste de cambio.")
                        
                        # Si no entró en ninguna específica (raro en churn), dar general
                        if not recos:
                             recos.append("- **Retención General**: Contactar proactivamente para entender necesidades actuales.")
                             
                        for r in recos:
                            st.markdown(r)
                            
                    else:
                        st.markdown("✅ **Diagnóstico: CLIENTE ESTABLE**")
                        st.markdown("El riesgo de fuga es bajo, pero se sugiere mantener el engagement:")
                        
                        recos_pos = []
                        if satisfaccion >= 4:
                            recos_pos.append("- **Promotor Potencial**: Alta satisfacción. Invitar a dejar una reseña a cambio de puntos.")
                        
                        if cashback > 200:
                             recos_pos.append("- **Upselling**: Tiene saldo alto de cashback. Sugerir productos de mayor valor para canje.")
                             
                        if antiguedad < 5:
                             recos_pos.append("- **Nurturing**: Cliente nuevo estable. Continuar secuencia de emails de bienvenida.")
                        
                        if not recos_pos:
                             recos_pos.append("- **Mantenimiento**: Continuar con newsletter mensual y ofertas estándar.")
                             
                        for r in recos_pos:
                            st.markdown(r)
            except Exception as e:
                st.error(f"Error procesando los datos: {e}")
                st.write("Verifica que las columnas coincidan con el modelo entrenado.")
        else:
            st.error("Error: Modelos no cargados correctamente. Ejecuta generate_models.py primero.")
