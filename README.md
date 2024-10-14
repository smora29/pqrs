# pqrs
## PQRS Calificación de Solicitudes
Este proyecto utiliza técnicas de Machine Learning para preprocesar, entrenar y evaluar un modelo de clasificación utilizando la biblioteca Kedro y herramientas de ML como scikit-learn y XGBoost. A continuación, se describe el flujo principal del proyecto.

Requisitos
Python 3.7 o superior
Bibliotecas necesarias:
pandas
scikit-learn
imblearn
xgboost
seaborn
matplotlib
Puedes instalar las dependencias ejecutando:

bash
Copiar código
pip install -r requirements.txt
Estructura del Proyecto
El flujo de trabajo se organiza en varias etapas que incluyen preprocesamiento de datos, creación de features, y entrenamiento de modelos. Aquí se describen las fases principales del pipeline:

0. Los insusmos y la informacion relvante al catalogo la pueden encontrar en el siguiente enlace donde aparecen los artefactos generados a lo largo de este desarrollo

https://drive.google.com/drive/folders/1OWVUijZjYDcEhE3NtvzkSP2mXyG-DSkR?usp=drive_link


1. Preprocesamiento de Datos
En esta fase, los datos son limpiados y transformados, con pasos como:

Imputación de valores faltantes
Conversión de cadenas a minúsculas
Estandarización de texto
Eliminación de duplicados

2. Ingeniería de Características
Algunas de las transformaciones clave incluyen:

Vectorización de texto usando TF-IDF
Cálculo de la longitud del texto
Análisis de sentimiento
Codificación One-Hot

3. Modelado
El pipeline incluye:

Muestreo balanceado: Se aplica RandomUnderSampler para manejar el desbalanceo de clases.
Entrenamiento del modelo: Utilizando XGBoost con búsqueda de hiperparámetros mediante RandomizedSearchCV.
Evaluación del modelo: Incluye métricas como classification_report.

4. Resultados
El modelo final genera predicciones y se analizan las características más importantes mediante la métrica de importancia de las características.

Los pipelines relacionados al los puntos mencionados en el apartado anterior tienen relacion en la siguiente ruta:
nlp/scr/pipelines/nombre_capa

Para su ejecucion es necesario activar el ambiente "source kedro-env-new/Scripts/activate"
ingresar a la carpeta del repositorio cd ruta/local 
en caso de que necesiten bajar la version actual de este repositorio lo pueden hacer atravez del comando " git pull origin development" desde su terminal 
para la ejecucion del pipeline de una capa lo hacen mediante el comando "kedro run --pipeline nombre_pipeline" ejemplo kedro run --pipeline data_processing


Los modulos CT ,CD y CI van configurados con estas instrucciones 




