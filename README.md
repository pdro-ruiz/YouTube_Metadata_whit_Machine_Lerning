# Análisis y modelado de metadatos de videos de YouTube

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg" alt="YouTube Logo">
</p>

## Descripción del Proyecto

Este proyecto se centra en el análisis y modelado de metadatos de videos de YouTube con el objetivo de resolver diversas problemáticas de negocio utilizando técnicas avanzadas de **Machine Learning** aprendidas durante la maestría. A través del procesamiento y exploración de los datos, se busca extraer información valiosa que permita abordar retos como la clasificación de categorías de videos, predicción de interacciones y recomendaciones personalizadas.

## Objetivos

- **Extracción y preprocesamiento** de datos de videos de YouTube provenientes de múltiples regiones.
- Desarrollo de **modelos de clasificación** para predecir la categoría de un video y determinar si los comentarios o calificaciones están deshabilitados.
- Implementación de **modelos de regresión** para predecir el número de "likes" y la ratio de "likes/dislikes".
- Aplicación de técnicas de **clusterización** para identificar grupos de videos similares.
- Creación de un **sistema de recomendación** que sugiera videos basados en la similitud con otros.

## Conclusiones del Análisis Exploratorio de Datos

Tras el trabajo realizado en el análisis de los datos, se han obtenido resultados relevantes que proporcionan una visión integral del comportamiento de las variables y su impacto en los modelos desarrollados:

1. **Análisis Exploratorio**:

   - **Concentración de videos**: La mayoría de los videos se encuentran en rangos específicos de "likes" y visualizaciones.
   - **Correlaciones positivas**: Se identificaron correlaciones significativas entre variables como duración y visualizaciones, comentarios y "likes".
   - **Gestión de valores atípicos**: Se detectaron y trataron valores atípicos para mejorar la robustez de los modelos.

2. **Preparación y Limpieza de Datos**:

   - **Imputación de valores nulos**: Se gestionaron valores nulos para garantizar la integridad de los datos.
   - **Transformaciones**: Se aplicaron transformaciones logarítmicas y normalizaciones para manejar la asimetría de los datos.
   - **Eliminación de registros problemáticos**: Se eliminaron registros mal formateados y categorías incompletas.

3. **Análisis de Datos**:

   - **Resumen estadístico**: Se realizó un resumen estadístico y un mapa de correlaciones para entender las relaciones entre variables.
   - **Análisis por categoría**: Se exploró el rendimiento y características de videos por categoría.
   - **Identificación de estrategias de contenido**: Se analizó cómo la longitud de títulos, descripciones y etiquetas influye en el rendimiento.

4. **Modelado**:

   - **Reto 1.1 - Clasificación de categorías**: Se optó por una Regresión Logística por su estabilidad y equilibrio entre rendimiento y generalización.
   - **Reto 1.2 - Clasificación de desactivación de comentarios**: El modelo MLP destacó por su rendimiento sobresaliente sin mostrar señales de sobreajuste.
   - **Reto 2.1 y 2.2 - Regresión**: LightGBM mostró una excelente capacidad para explicar la variabilidad de los datos en la predicción del número de "likes" y la ratio "likes/dislikes".
   - **Reto 3 - Clusterización**: DBSCAN ofreció el mejor desempeño en la identificación de grupos en los datos.
   - **Reto 4 - Recomendación**: El sistema basado en KNN demostró buena efectividad al generar recomendaciones con alta similitud.

## Estructura del Proyecto

La estructura del proyecto es la siguiente:

```
proyecto/
├── exploratory_analysis.ipynb
├── train.conf
├── preprocess_data.py
├── train_models.py
├── inference_models.py
├── logger/
│     └── logger.py
├── logs/
├── main.py
├── data/
│     ├── eda/
│     │     ├── cleaned/
│     │     ├── pre-processed/
│     │     ├──  processed/
│     │     ├── translate/
│     │     └── vectorized/
│     ├── industrialized/
│     │         ├──  pre_processed/
│     │         └── processed/
│     │                 ├── pkl/
│     │                 └── vectorizers/
│     └── raw_data/
└── models/
    ├── classification_categories/
    │             ├── data/
    │             ├── weight/
    │             ├── metrics/
    │             ├──  results/
    │             └── logistic_regression.py
    ├── classification_video_disabled/
    │             ├── data/
    │             ├── weight/
    │             ├── metrics/
    │             ├── results/
    │             └──  mlp.py
    ├── regression_like_ratio/
    │             ├── data/
    │             ├── weight/
    │             ├── metrics/
    │             ├── results/
    │             └── lightgbm_regressor.py
    ├── regression_number_of_likes/
    │             ├── data/
    │             ├── weight/
    │             ├── metrics/
    │             ├── results/
    │             └── lightgbm_regressor.py
    ├── clusterization/
    │             ├── weight/
    │             ├── metrics/
    │             ├── results/
    │             ├── pca_clusterization.py
    │             └── dbscan_clusterization.py
    └── recommendation/
                  ├── weight/
                  ├── metrics/
                  ├── results/
                  ├── knn_recommend.py
                  └── knn_recommender.py
```

### Descripción de los directorios y archivos principales

- **`exploratory_analysis.ipynb`**: Notebook de Jupyter donde se realiza el análisis exploratorio de datos y pruebas de diferentes modelos.
- **`train.conf`**: Archivo de configuración que contiene rutas, hiperparámetros y ajustes para los modelos.
- **`preprocess_data.py`**: Script para preprocesar y limpiar los datos en crudo.
- **`train_models.py`**: Script que entrena los modelos especificados y guarda sus pesos.
- **`inference_models.py`**: Script que realiza la inferencia utilizando los modelos entrenados.
- **`logger/`**: Directorio que contiene el módulo para el manejo de logs.
- **`logs/`**: Directorio donde se almacenan los registros de ejecución.
- **`data/`**: Carpeta que contiene los datos en sus diferentes estados (crudo, preprocesado, procesado).
- **`models/`**: Contiene los distintos modelos, sus pesos, métricas y resultados.

## Dataset

El dataset utilizado incluye varios meses de datos sobre los videos de tendencias diarias de YouTube. Los datos contienen información detallada de cada video, como su categoría, número de vistas, likes, dislikes, comentarios, entre otros.

- **Enlace al dataset**: [Dataset metadatos de YouTube](https://drive.google.com/file/d/1-ZvMZzTZCmUlfVU8po0-nuBGL_LBDmTN/view?usp=sharing)

## Explicación de los scripts y cómo ejecutarlos

### Requisitos previos

- Python 3.7 o superior.
- Bibliotecas listadas en `requirements.txt` (asegúrese de instalar todas las dependencias).

### Preprocesamiento de Datos

1. **Preprocesar los datos crudos**:

   Ejecute el script `preprocess_data.py` pasando como argumentos la ruta de los datos en crudo y la ruta donde se guardarán los datos procesados.

   ```bash
   python preprocess_data.py --input_dir data/raw_data --output_dir data/industrialized
   ```

2. **Descripción del proceso**:

   - **Carga y preparación**: Lee los archivos CSV y JSON de las diferentes regiones, prepara y codifica los archivos para su procesamiento.
   - **Limpieza de datos**: Elimina registros inválidos y categorías no deseadas.
   - **Transformación de columnas de texto**: Normaliza texto, extrae emojis, URLs y realiza limpieza de caracteres especiales.
   - **Ingeniería de características**: Crea nuevas características como codificaciones cíclicas de fechas, ratios y transformaciones logarítmicas.
   - **Análisis y eliminación de duplicados**: Identifica y elimina videos duplicados para evitar sesgos en los modelos.
   - **Vectorización de texto**: Vectoriza las columnas de texto utilizando TF-IDF y elimina palabras vacías en múltiples idiomas.
   - **Selializacion**

### Entrenamiento de modelos

1. **Configurar los hiperparámetros y rutas**:

   Modifique el archivo `train.conf` según sea necesario, especificando rutas a los datos procesados, hiperparámetros de los modelos y rutas donde se guardarán los modelos entrenados.

2. **Entrenar los modelos**:

   Ejecute el script `train_models.py` pasando el archivo de configuración como argumento.

   ```bash
   python train_models.py train.conf
   ```

3. **Descripción del proceso**:

   - **Carga de datos**: Importa los datos preprocesados necesarios para el entrenamiento.
   - **Entrenamiento**: Entrena los modelos especificados en el archivo de configuración.
   - **Guardado de modelos**: Almacena los modelos entrenados y sus pesos en las rutas especificadas.

###  Inferencia con modelos entrenados

1. **Realizar inferencia**:

   Ejecute el script `inference_models.py` pasando el archivo de configuración como argumento.

   ```bash
   python inference_models.py train.conf
   ```

2. **Descripción del proceso**:

   - **Carga de modelos y datos**: Carga los modelos entrenados y los conjuntos de datos de prueba.
   - **Predicciones**: Realiza predicciones utilizando los modelos cargados.
   - **Evaluación**: Calcula métricas de rendimiento y guarda los resultados.

### Recomendación de videos

1. **Obtener recomendaciones**:

   Ejecute el script `knn_recommend.py` especificando el archivo de configuración.

   ```bash
   python knn_recommend.py --config_path train.conf
   ```

2. **Descripción del proceso**:

   - **Interacción con el usuario**: Solicita al usuario el ID o índice de un video para el cual se desean obtener recomendaciones.
   - **Generación de recomendaciones**: Utiliza el modelo KNN entrenado para encontrar videos similares.
   - **Presentación de resultados**: Muestra y guarda las recomendaciones obtenidas.

## Aprendizaje Obtenido

Durante el desarrollo de este proyecto, hemos adquirido valiosos aprendizajes que han enriquecido nuestra experiencia en el campo del Machine Learning:

- **Desafíos en el procesamiento de texto**: Nos esforzamos intensamente en limpiar y normalizar los textos de títulos, descripciones y etiquetas. Intentamos implementar soluciones económicas para traducir el contenido multilingüe, pero el tiempo limitado nos impidió profundizar más en esta área. Este reto nos enseñó la importancia de planificar adecuadamente las tareas y priorizar según los recursos disponibles.

- **Muestreo en Big Data para EDA**: Aprendimos sobre la necesidad de utilizar muestras representativas de datos al realizar análisis exploratorios en conjuntos de datos grandes. Esto nos habría ahorrado tiempo y recursos computacionales, permitiéndonos obtener insights valiosos de manera más eficiente.

- **Utilidad de la computación en la nube**: Reconocimos las ventajas de la computación en la nube para proyectos de gran escala. La capacidad de desplegar múltiples máquinas y distribuir procesos evita cuellos de botella asociados con la centralización en una única máquina local. En futuros proyectos, consideraremos esta opción para mejorar la escalabilidad y eficiencia.

- **Camino hacia la especialización**: Con humildad, reconocemos que aún tenemos mucho por aprender para convertirnos en expertos en la materia. Este proyecto ha sido un paso significativo, pero también nos ha mostrado la amplitud y profundidad del campo del Machine Learning y la importancia de la formación continua.

## Autor

**Pedro Ruiz**

- **LinkedIn**: [linkedin.com/in/pdro-ruiz](https://linkedin.com/in/pdro-ruiz/)

## Licencia

Este proyecto está bajo la Licencia MIT. Consulte el archivo [LICENSE](LICENSE) para obtener más información.

## Agradecimientos

A todos los profesores por su paciencia y dedicación, y los compañeros de la maestría por su apoyo.
