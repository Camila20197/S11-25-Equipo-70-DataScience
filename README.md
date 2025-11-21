# S11-25-Equipo-70-DataScience

## Descripción del Proyecto

Este proyecto de Data Science se enfoca en el análisis y predicción del **Customer Churn** (abandono de clientes) en el sector retail online. El objetivo principal es identificar patrones de comportamiento que permitan predecir qué clientes tienen mayor probabilidad de abandonar el servicio, permitiendo implementar estrategias de retención proactivas.

## Estructura del Proyecto

```
S11-25-Equipo-70-DataScience/
│
├── datos/
│   ├── online_retail_customer_churn.csv    # Dataset original (1000 registros)
│   └── dataset_retail_limpio.csv           # Dataset procesado y limpio
│
├── EDA/
│   └── etapa_EDA.ipynb                     # Notebook con análisis exploratorio
│
├── image.png                                # Matriz de correlaciones
└── README.md                                # Documentación del proyecto
```

## Dataset

### Información General
- **Tamaño**: 1,000 registros de clientes
- **Fuente**: `online_retail_customer_churn.csv`
- **Valores nulos**: No se encontraron valores nulos en el dataset
- **Duplicados**: No se encontraron registros duplicados

### Variables del Dataset

#### Variables Numéricas

| Variable | Descripción | Transformaciones Aplicadas |
|----------|-------------|---------------------------|
| `Edad` | Edad del cliente | - |
| `Ingreso_Anual` | Ingresos anuales del cliente | Escalado x1000 |
| `Total_Gastado` | Monto total gastado por el cliente | - |
| `Numeros_Compras` | Cantidad total de compras realizadas | - |
| `Importe_Promedio_Transaccion` | Valor promedio por transacción | - |
| `Numero_de_Devoluciones` | Cantidad de productos devueltos | - |
| `Cantidad_de_Contactos_Soporte` | Número de contactos con soporte | - |
| `Anios_como_cliente` | Años de antigüedad como cliente | - |
| `Ultimo_dia_compra` | Días transcurridos desde la última compra | - |
| `Nivel_Satisfaccion` | Puntuación de satisfacción del cliente | - |

#### Variables Categóricas

| Variable | Descripción | Valores | Transformaciones |
|----------|-------------|---------|------------------|
| `Genero` | Género del cliente | Male, Female, Other | Convertido a minúsculas |
| `Esta_Suscripto_via_Email` | Suscripción a emails | True/False | Convertido a int (0/1) |
| `Respuesta_a_Promocion` | Respuesta a promociones | Responded/Ignored | Convertido a minúsculas |
| `Target` | Variable objetivo - Churn | True/False | Convertido a int (0/1) |

## Análisis Exploratorio de Datos (EDA)

### Librerías Utilizadas
- **NumPy**: Operaciones numéricas
- **Pandas**: Manipulación y análisis de datos
- **Matplotlib**: Visualizaciones básicas
- **Seaborn**: Visualizaciones estadísticas avanzadas

### Proceso de Limpieza de Datos

1. **Renombramiento de Variables**
   - Todas las variables fueron traducidas al español para mejor comprensión
   - Se mantuvieron nombres descriptivos y claros

2. **Conversión de Tipos de Datos**
   - Variables categóricas convertidas a string y normalizadas a minúsculas
   - Variables booleanas convertidas a enteros (0/1)

3. **Escalado de Variables**
   - `Ingreso_Anual` multiplicado por 1000 para representar valores reales en unidades monetarias

4. **Validación de Calidad**
   - Verificación de valores nulos: ✅ Sin valores nulos
   - Verificación de duplicados: ✅ Sin duplicados

### Análisis Univariado

Se realizó análisis de las siguientes variables de comportamiento clave:
- Número de compras
- Importe promedio de transacción
- Días desde última compra
- Contactos con soporte
- Número de devoluciones
- Total gastado
- Años como cliente
- Suscripción vía email
- Target (Churn)

**Visualizaciones generadas:**
- Histogramas con KDE para distribuciones
- Boxplots para identificación de outliers
- Gráficos de conteo para variables categóricas

### Análisis de Correlaciones

Se generó una matriz de correlaciones para identificar relaciones entre variables de comportamiento y la variable objetivo (Target).

![Matriz de Correlaciones](image.png)

**Variables con mayor correlación al Churn:**
- Comportamiento de compra
- Satisfacción del cliente
- Interacciones con soporte
- Frecuencia de devoluciones

## Tecnologías y Herramientas

- **Python 3.x**
- **Jupyter Notebook**
- **Pandas** - Manipulación de datos
- **NumPy** - Cálculos numéricos
- **Matplotlib** - Visualizaciones
- **Seaborn** - Visualizaciones estadísticas

## Resultados Principales

1. **Dataset Limpio**: Se generó un dataset procesado (`dataset_retail_limpio.csv`) listo para modelado
2. **Variables Identificadas**: Se identificaron las variables más relevantes para predicción de churn
3. **Insights de Comportamiento**: Se documentaron patrones de comportamiento asociados al abandono de clientes
4. **Visualizaciones**: Se crearon visualizaciones para entender distribuciones y correlaciones

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

- El dataset limpio se encuentra comentado en el notebook para evitar sobrescrituras accidentales
- Se recomienda revisar el notebook `etapa_EDA.ipynb` para análisis detallado
- Las transformaciones aplicadas son reversibles para análisis posteriores

---

*Última actualización: Noviembre 2025*