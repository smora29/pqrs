"""
This is a boilerplate pipeline 'model_output'
generated using Kedro 0.18.14
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as imbPipeline
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
import numpy as np
import logging


import src.bbog_gd_seguros_churn_ml.pipelines.data_processing.nodes as processing
import src.bbog_gd_seguros_churn_ml.pipelines.primary.nodes as primary
import src.bbog_gd_seguros_churn_ml.pipelines.feature.nodes as feature
import src.bbog_gd_seguros_churn_ml.pipelines.model_input.nodes as model

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def model_output(df, importances_df, params, model_search):
    """
    Función para procesar datos, seleccionar características más importantes y generar predicciones.

    Entradas:
    df: DataFrame original a procesar.
    importances_df: DataFrame con las importancias de las características.
    params: Diccionario con parámetros, como columnas objetivo, ID y número de características.
    model_search: Modelo entrenado para hacer predicciones.

    Retorna:
    df_filtrado: DataFrame con predicciones y la columna 'complaint id'.
    """
    
    # Extraer parámetros clave
    id_col = params['id']
    target = params['target']
    top_n = params['top_features']
    
    # Preprocesamiento
    df = processing.convertir_a_minusculas(df, params)
    df = processing.standardize_strings(df, params)
    df = processing.values_to_null(df)
    
    # Primary
    df = primary.remove_duplicates(df)
    df = primary.clean_dataframe_by_missing_values(df, params)
    df = primary.impute_missing_values(df)
    df = primary.create_effective_response(df)
    
    # Feature Engineering
    df = feature.create_interactions(df)
    df = feature.encode_one_hot(df, params['column_product'])
    df = feature.calculate_text_length(df, params['column_issue'])
    df = feature.calculate_sentiment(df, params['column_response'])
    df = feature.add_temporal_features(df)
    
    # Seleccionar las 'top_n' características más importantes
    top_features = importances_df.head(top_n)['Feature'].values
    df_filtrado = df[top_features]
    
    # Predicciones
    df_filtrado['respuesta'] = model_search.predict(df_filtrado)
    
    # Añadir columna 'complaint id'
    df_filtrado['complaint id'] = df[id_col]
    
    return df_filtrado
