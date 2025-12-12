#  Dashboard de Predicción de Churn - E-commerce

Este proyecto implementa un dashboard interactivo para el análisis y predicción de abandono de clientes (Churn) en una plataforma de E-commerce. Utiliza un modelo de Machine Learning (KNN) para identificar clientes en riesgo y propone estrategias de retención.

##  Estructura del Proyecto

El directorio `dashboard/` contiene los siguientes archivos clave:

*   `app.py`: La aplicación principal de Streamlit.
*   `generate_models.py`: Script para entrenar y regenerar el modelo predictivo.
*   `logo.png`: Logotipo del proyecto.
*   `requirements.txt`: Lista de dependencias necesarias.
*   `*.pkl`: Archivos del modelo entrenado (se generan automáticamente).

##  Requisitos e Instalación

1.  **Requisito previo**: Tener Python instalado (versión 3.8 o superior).
2.  **Instalar dependencias**:
    Abre tu terminal en la carpeta `dashboard` y ejecuta:

    ```bash
    pip install -r requirements.txt
    ```

##  Cómo Correr el Dashboard

Una vez instaladas las dependencias, ejecuta el siguiente comando dentro de la carpeta `dashboard`:

```bash
streamlit run app.py
```

El dashboard se abrirá automáticamente en tu navegador predeterminado (usualmente en `http://localhost:8501`).

##  Regenerar el Modelo (Opcional)

Si necesitas actualizar el modelo o si los archivos `.pkl` no existen, ejecuta:

```bash
python generate_models.py
```

Este script leerá los datos desde `../datos/dataset_ecommerce_limpio.csv`, entrenará el modelo KNN, optimizará el umbral de decisión y guardará los nuevos archivos del modelo.

##  Secciones del Dashboard

1.  **Análisis Exploratorio (EDA)**: Visualización de datos y correlaciones.
2.  **Definición del Churn**: Explicación de las reglas de negocio para etiquetar el abandono.
3.  **Modelado Predictivo**: Métricas del modelo (Accuracy, ROC-AUC, Matriz de Confusión).
4.  **Segmentación de Clientes**: Análisis de riesgo por grupos (Antigüedad, Satisfacción, etc.).
5.  **Dashboard Analítico**: KPIs principales y resumen ejecutivo.
6.  **Recomendaciones de Acción**: Estrategias de negocio basadas en los hallazgos.
