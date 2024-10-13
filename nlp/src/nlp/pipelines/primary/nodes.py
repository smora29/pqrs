"""
This is a boilerplate pipeline 'primary'
generated using Kedro 0.18.14
"""
from typing import Dict, List, Any, Tuple

import re
import pandas as pd
import numpy as np
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 1 Validar duplicados 
def remove_duplicates(df):
    """
    Elimina filas duplicadas en el DataFrame basado en las columnas 'Date received', 'Issue', y 'Complaint ID'.
    
    Parámetros:
    df (pd.DataFrame): El DataFrame de entrada que contiene los datos.
    
    Retorna:
    pd.DataFrame: El DataFrame sin filas duplicadas.
    """
    # Eliminar duplicados basados en las columnas 'Date received', 'Issue' y 'Complaint ID'
    df_no_duplicates = df.drop_duplicates(subset=['date received', 'issue', 'complaint id'])
    
    return df_no_duplicates


# 2. tratamienito valores missing 

def clean_dataframe_by_missing_values(df, parameters):
    """
    Elimina las columnas con un porcentaje de valores nulos superior al umbral proporcionado.

    Parámetros:
    df (pd.DataFrame): El DataFrame de entrada que contiene los datos.
    threshold (int): El porcentaje de valores nulos que, si es superado por una columna, 
                     causará que esa columna sea eliminada (valor por defecto es 70).

    Retorna:
    pd.DataFrame: Un DataFrame limpio con las columnas que tienen menos del umbral de valores nulos.
    """
    threshold= parameters['threshold']
    # Calcular el porcentaje de valores nulos por columna
    missing_percentage = df.isnull().mean() * 100
    
    # Identificar las columnas que superan el umbral
    columns_to_drop = missing_percentage[missing_percentage > 70].index

    # Eliminar las columnas que tienen más del umbral de valores nulos
    df_cleaned = df.drop(columns=columns_to_drop)
    
    return df_cleaned

# 3  Imputacion 

def impute_missing_values(df):
    """
    Imputa valores faltantes en el DataFrame según el porcentaje de valores nulos en las columnas:
    - Columnas con menos del 30% de valores nulos: se imputan con la moda.
    - Columnas con entre el 30% y el 69% de valores nulos: se imputan con la palabra 'Other'.
    
    Parámetros:
    df (pd.DataFrame): El DataFrame de entrada que contiene los datos.
    
    Retorna:
    pd.DataFrame: El DataFrame con los valores imputados.
    """
    # Calcular el porcentaje de valores nulos por columna
    missing_percentage = df.isnull().mean() * 100
    
    # Columnas con menos del 30% de valores nulos
    columns_mode = missing_percentage[missing_percentage < 30].index

    # Columnas con entre el 30% y el 69% de valores nulos
    columns_unknown = missing_percentage[(missing_percentage >= 30) & (missing_percentage <= 69)].index
    
    # Imputar las columnas con menos del 30% con la moda
    for column in columns_mode:
        mode_value = df[column].mode()[0]  # Obtener la moda de la columna
        df[column].fillna(mode_value, inplace=True)
    
    # Imputar las columnas con entre el 30% y el 69% con la palabra 'Other'
    df[columns_unknown] = df[columns_unknown].fillna('Other')
    
    return df

# 4 _TArget
# Nodo para generar la variable objetivo en Kedro
def create_effective_response(df: pd.DataFrame) -> pd.DataFrame:
    df['effective_response'] = ((df['timely response?'] == 'yes') & 
                                (df['consumer disputed?'] == 'no')).astype(int)
    return df
