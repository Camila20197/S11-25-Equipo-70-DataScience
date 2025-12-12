# Documentación del Proyecto: Predicción de Fuga de Clientes (ChurnZero)

## 1. Visión General
Este proyecto tiene como objetivo **reducir la pérdida de clientes** (churn) en la plataforma de e-commerce mediante el uso de inteligencia artificial. Hemos desarrollado un sistema que no solo identifica quién se va a ir, sino que explica *por qué* y sugiere *qué hacer* para retenerlo.

### Problema Detectado
- Tasa de Fuga Histórica: **16.8% - 17.1%**
- Impacto: Pérdida significativa de ingresos recurrentes.
- Hallazgo Principal: La fuga ocurre mayoritariamente en los primeros meses (problema de onboarding) y está fuertemente ligada a quejas no resueltas.

---

## 2. Metodología de Ciencia de Datos

Seguimos un proceso riguroso de 6 etapas para asegurar la calidad de los resultados:

### Paso 1: Entendimiento del Negocio
Definimos el "Churn" no solo como la baja del servicio, sino también detectando a clientes que dejan de comprar abruptamente (abandono silencioso).

### Paso 2: Análisis Exploratorio (EDA)
Analizamos más de 3,900 registros históricos.
- **Limpieza**: Se trataron valores nulos en columnas como `Antiguedad` y `Dias_Ultima_Compra`.
- **Insights**: Descubrimos que el "Cashback" es un fuerte retenedor y que la satisfacción por debajo de 3 estrellas dispara el riesgo.

### Paso 3: Definición del Target
Creamos una variable objetivo clara (`Target = 1`) para entrenar a la IA, diferenciando entre clientes leales y fugados.

### Paso 4: Modelado Predictivo
Evaluamos múltiples algoritmos:
- **K-Nearest Neighbors (KNN)**: Seleccionado como modelo final.
    - **Por qué ganó**: Mejor capacidad para detectar fugas reales (Recall ~88%). Es decir, prefiere pecar de precavido y alertar un riesgo falso antes que dejar pasar una fuga real.
- **XGBoost**: Evaluado como alternativa, con mayor precisión global pero menor sensibilidad para detectar casos críticos.

### Paso 5: Segmentación
Agrupamos a los clientes en niveles de riesgo (Alto, Medio, Bajo) para priorizar recursos. No tratamos igual a todos: nos enfocamos en el 10% más crítico.

### Paso 6: Dashboard y Acción
Desarrollamos la herramienta interactiva que traduce estos modelos matemáticos en pantallas de negocio.

---

## 3. ¿Cómo usar el Dashboard?

La herramienta está dividida en 7 secciones estratégicas:

1.  **Exploración de Datos (EDA)**: Para analistas que quieran ver los datos crudos y estadísticas.
2.  **Definición del Churn**: Explica las reglas de negocio usadas.
3.  **Modelado Predictivo**: Muestra las métricas técnicas de la IA (Accuracy, F1-Score).
4.  **Segmentación**: Muestra cómo se dividen los grupos de riesgo.
5.  **Dashboard Analítico**: **[VISTA EJECUTIVA]** Resumen gráfico con KPIs de negocio, impacto monetario y explicaciones visuales sencillas.
6.  **Recomendaciones**: El "Plan de Acción". Qué estrategias implementar mañana mismo.
7.  **Predicción de Fuga (Simulador)**: **[HERRAMIENTA OPERATIVA]**
    - Ingrese los datos de un cliente individual.
    - El sistema le dirá su % de probabilidad de fuga.
    - El sistema le dará 3-5 recomendaciones personalizadas basadas en *ese* cliente específico.

---

## 4. Requerimientos Técnicos
Para ejecutar este proyecto en otro entorno:

- **Lenguaje**: Python 3.9+
- **Librerías Clave**:
    - `streamlit`: Interfaz gráfica
    - `scikit-learn`: Motor de IA (KNN)
    - `pandas` / `numpy`: Procesamiento de datos
    - `matplotlib` / `seaborn`: Gráficos
- **Archivos del Modelo**:
    - `modelo_knn_churn_final.pkl`: El "cerebro" entrenado.
    - `scaler_knn_churn.pkl`: Para normalizar los datos nuevos.

## Autoría
**Equipo de Data Science - Equipo 70**
S11-25 - No Country
