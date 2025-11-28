# S11-25-Equipo-70-DataScience

## Descripción del Proyecto

Este proyecto de Data Science se enfoca en el análisis y predicción del **Customer Churn** (abandono de clientes) en el sector de e-commerce. El objetivo principal es identificar patrones de comportamiento que permitan predecir qué clientes tienen mayor probabilidad de abandonar el servicio, permitiendo implementar estrategias de retención proactivas.

## Estructura del Proyecto

```
S11-25-Equipo-70-DataScience/
│
├── datos/
│   ├── data_ecommerce_customer_churn.csv   # Dataset original
│   └── dataset_ecommerce_limpio.csv        # Dataset procesado y limpio
│
├── EDA/
│   └── etapa_EDA_segundoDataset.ipynb      # Notebook con análisis exploratorio
│
├── final_model.sav                          # Modelo entrenado
└── README.md                                # Documentación del proyecto
```

## Dataset

### Información General
- **Tamaño**: Dataset de clientes de e-commerce
- **Fuente**: `data_ecommerce_customer_churn.csv`
- **Valores nulos**: Se encontraron valores nulos en Antiguedad, Distancia_Almacen y Dias_Ultima_Compra
- **Duplicados**: No se encontraron registros duplicados

### Variables del Dataset

#### Variables Numéricas

| Variable | Descripción | Transformaciones Aplicadas |
|----------|-------------|---------------------------|
| `Antiguedad` (Tenure) | Tiempo como cliente | Imputación de nulos con mediana |
| `Distancia_Almacen` (WarehouseToHome) | Distancia del almacén al hogar | Imputación de nulos con mediana |
| `Numero_Dispositivos` (NumberOfDeviceRegistered) | Cantidad de dispositivos registrados | - |
| `Nivel_Satisfaccion` (SatisfactionScore) | Puntuación de satisfacción del cliente | - |
| `Numero_Direcciones` (NumberOfAddress) | Cantidad de direcciones registradas | - |
| `Dias_Ultima_Compra` (DaySinceLastOrder) | Días transcurridos desde la última compra | Imputación de nulos con mediana |
| `Monto_Cashback` (CashbackAmount) | Monto de cashback recibido | - |

#### Variables Categóricas

| Variable | Descripción | Valores | Transformaciones |
|----------|-------------|---------|------------------|
| `Estado_Civil` (MaritalStatus) | Estado civil del cliente | Variados | Convertido a string y minúsculas |
| `Categoria_Preferida` (PreferedOrderCat) | Categoría de producto preferida | Variadas categorías | Convertido a string y minúsculas |
| `Queja` (Complain) | Si el cliente presentó quejas | 0/1 | Convertido a int |
| `Target` (Churn) | Variable objetivo - Abandono | 0/1 | Convertido a int |

## Análisis Exploratorio de Datos (EDA)

### Librerías Utilizadas
- **NumPy**: Operaciones numéricas
- **Pandas**: Manipulación y análisis de datos
- **Matplotlib**: Visualizaciones básicas
- **Seaborn**: Visualizaciones estadísticas avanzadas

### Proceso de Limpieza de Datos

1. **Renombramiento de Variables**
   - Tenure → Antiguedad
   - WarehouseToHome → Distancia_Almacen
   - NumberOfDeviceRegistered → Numero_Dispositivos
   - PreferedOrderCat → Categoria_Preferida
   - SatisfactionScore → Nivel_Satisfaccion
   - MaritalStatus → Estado_Civil
   - NumberOfAddress → Numero_Direcciones
   - Complain → Queja
   - DaySinceLastOrder → Dias_Ultima_Compra
   - CashbackAmount → Monto_Cashback
   - Churn → Target

2. **Conversión de Tipos de Datos**
   - Variables categóricas (Estado_Civil, Categoria_Preferida) convertidas a string y normalizadas a minúsculas
   - Variables binarias (Queja, Target) convertidas a enteros (0/1)

3. **Tratamiento de Valores Nulos**
   - `Antiguedad`: Imputación con mediana
   - `Distancia_Almacen`: Imputación con mediana
   - `Dias_Ultima_Compra`: Imputación con mediana
   - Se utilizó la mediana por ser menos sensible a outliers

4. **Validación de Calidad**
   - Verificación de valores nulos: Imputados correctamente
   - Verificación de duplicados: Sin duplicados

### Análisis Univariado

Se realizó análisis de las siguientes variables de comportamiento clave:
- Antiguedad
- Distancia_Almacen
- Numero_Dispositivos
- Nivel_Satisfaccion
- Numero_Direcciones
- Queja
- Dias_Ultima_Compra
- Monto_Cashback
- Target (Churn)

**Visualizaciones generadas:**
- Histogramas con KDE para distribuciones de variables numéricas
- Boxplots para identificación de outliers
- Gráficos de conteo para variables categóricas (Categoria_Preferida, Estado_Civil)

### Pruebas de Normalidad
Se aplicó la prueba de Kolmogorov-Smirnov para evaluar la distribución de las variables numéricas.
- **Resultado**: Todas las variables numéricas mostraron un p-valor < 0.05.
- **Conclusión**: Se rechaza la hipótesis nula de normalidad. Ninguna variable numérica sigue una distribución normal, lo que sugiere el uso de modelos no paramétricos o transformaciones previas.

### Detección de Outliers
Se analizaron los valores atípicos utilizando el Rango Intercuartílico (IQR) y Isolation Forest.
- **Numero_Dispositivos**: 6.8% de outliers.
- **Monto_Cashback**: 8% de outliers.
- **Distancia_Almacen**: 0.35% de outliers.
- **Dias_Ultima_Compra**: 0.5% de outliers.
- **Multivariado**: Se detectaron 15 outliers multivariados mediante Isolation Forest.

### Análisis de Correlaciones

Se generó una matriz de correlaciones para identificar relaciones entre variables de comportamiento y la variable objetivo (Target).

**Variables con mayor correlación positiva al Churn:**
1. **Queja** (0.25): Los clientes que presentan quejas tienen significativamente mayor probabilidad de abandonar
2. **Dias_Ultima_Compra** (0.09): Mayor tiempo sin comprar aumenta el riesgo de churn

**Variables con correlación negativa al Churn:**
1. **Antiguedad** (-0.35): Clientes nuevos (baja antigüedad) tienen mayor riesgo de churn
2. **Nivel_Satisfaccion** (-0.03): Menor satisfacción está asociada con mayor churn
3. **Monto_Cashback** (-0.16): Menor cashback recibido se asocia con mayor abandono

### Análisis de Componentes Principales (PCA)
Se realizó una reducción de dimensionalidad a 2 componentes para visualizar la separabilidad de las clases.
- **Varianza Explicada**: Las 2 primeras componentes explican aproximadamente el 50% de la varianza.
- **Observación**: Las clases "Churn" y "No Churn" se encuentran mezcladas en el espacio proyectado.
- **Conclusión**: El problema no es linealmente separable en bajas dimensiones, lo que descarta modelos lineales simples sin ingeniería de características avanzada.

## Tecnologías y Herramientas

- **Python 3.x**
- **Jupyter Notebook**
- **Pandas** - Manipulación de datos
- **NumPy** - Cálculos numéricos
- **Matplotlib** - Visualizaciones
- **Seaborn** - Visualizaciones estadísticas

## Análisis Bivariado

### Churn por Variables Categóricas

1. **Distribución de Churn**
   - Se generaron gráficos de conteo y porcentaje de churn
   - Análisis de tasa de churn por categoría preferida
   - Análisis de tasa de churn por estado civil

2. **Impacto de Quejas**
   - Los clientes con quejas muestran una tasa de churn significativamente mayor
   - Se generaron visualizaciones comparativas de churn según presencia de quejas

### Churn por Variables Numéricas

1. **Boxplots Comparativos**
   - Se compararon todas las variables numéricas según la presencia de churn
   - Variables analizadas: Antiguedad, Distancia_Almacen, Numero_Dispositivos, Nivel_Satisfaccion, Dias_Ultima_Compra, Monto_Cashback

2. **Distribuciones Superpuestas**
   - Antigüedad: Clientes con menor antigüedad muestran mayor churn
   - Nivel de Satisfacción: Distribución más baja en clientes que abandonan
   - Días desde última compra: Mayor inactividad correlaciona con churn

3. **Estadísticas Descriptivas**
   - Se calcularon estadísticas por grupo de churn para identificar diferencias significativas

## Principales Hallazgos

### Variables más Correlacionadas con Churn:
1. **Queja**: Los clientes que presentan quejas tienen mayor probabilidad de abandonar
2. **Antigüedad**: Clientes nuevos (baja antigüedad) tienen mayor riesgo de churn
3. **Días desde última compra**: Mayor tiempo sin comprar aumenta el riesgo
4. **Nivel de Satisfacción**: Menor satisfacción está asociada con mayor churn
5. **Monto Cashback**: Menor cashback recibido se asocia con mayor abandono

### Recomendaciones para el Modelo:
- Priorizar variables: Queja, Antiguedad, Nivel_Satisfaccion, Dias_Ultima_Compra
- Considerar ingeniería de features con Categoria_Preferida y Estado_Civil
- Evaluar balanceo de clases para Target
- Analizar interacciones entre Queja y otras variables de satisfacción

## Resultados Principales

1. **Dataset Limpio**: Se generó un dataset procesado (`dataset_ecommerce_limpio.csv`) listo para modelado
2. **Variables Identificadas**: Se identificaron las variables más relevantes para predicción de churn
3. **Insights de Comportamiento**: Se documentaron patrones de comportamiento asociados al abandono de clientes
4. **Visualizaciones Completas**: 
   - Análisis univariado de todas las variables
   - Análisis bivariado de variables vs churn
   - Matriz de correlaciones
   - Distribuciones superpuestas por grupo de churn

## Recomendación de Modelos
Basado en el EDA (no normalidad, presencia de outliers, no linealidad), se sugieren los siguientes modelos:

### Recomendados
1. **XGBoost / LightGBM**: Manejan bien datos no lineales, outliers y valores faltantes. Alto rendimiento en datos tabulares.
2. **Random Forest**: Robusto ante outliers y no requiere normalización. Captura relaciones no lineales.
3. **KNN (K-Nearest Neighbors)**: No asume normalidad (no paramétrico). Requiere escalado de datos.
4. **SVM (Support Vector Machines)**: Con kernel RBF para manejar la no linealidad.

### No Recomendados
- **Regresión Logística / LDA / Naive Bayes**: Asumen normalidad o separabilidad lineal, supuestos que no se cumplen en este dataset.

## Próximos Pasos

1. **Modelado Predictivo**
   - Implementar modelos de Machine Learning (Logistic Regression, Random Forest, XGBoost)
   - Realizar validación cruzada
   - Optimizar hiperparámetros

2. **Feature Engineering**
   - Crear nuevas variables derivadas
   - Aplicar técnicas de encoding para variables categóricas
   - Normalizar/estandarizar variables numéricas

3. **Evaluación de Modelos**
   - Métricas: Accuracy, Precision, Recall, F1-Score, AUC-ROC
   - Matriz de confusión
   - Análisis de importancia de variables

## Equipo

**Equipo 70 - Data Science**  
NoCountry - S11-25

## Notas Adicionales

- El dataset limpio se guardó como `dataset_ecommerce_limpio.csv` en la carpeta datos/
- Se recomienda revisar el notebook `etapa_EDA_segundoDataset.ipynb` para análisis detallado y visualizaciones
- Las transformaciones aplicadas son reversibles para análisis posteriores
- Se utilizó imputación con mediana para valores nulos por su robustez ante outliers

---

*Última actualización: Noviembre 2025*