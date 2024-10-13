"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.9
"""
"""
Nodos de la capa raw
"""
from typing import Dict, List, Any, Tuple

import re
import pandas as pd
import numpy as np
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 1 . minuscalas 
def convertir_a_minusculas(df, columna_excluida='Complaint ID'):
    # Convertir los nombres de las columnas a minúsculas, excepto la columna excluida
    df.columns = [col.lower() if col != columna_excluida else col for col in df.columns]
    
    # Convertir a minúsculas las columnas que sean de tipo object (string), excepto la columna excluida
    for col in df.select_dtypes(include=['object']).columns:
        if col != columna_excluida:
            df[col] = df[col].str.lower()
    
    return df

# 2. Valores especiales 
def standardize_strings(df: pd.DataFrame, param_id_col: str) -> pd.DataFrame:
    """
    Estandariza los strings para columnas de texto en un DataFrame de Pandas.
    Reemplaza caracteres con tilde por sus versiones sin tilde, y otros caracteres
    especiales los reemplaza por "_".
    
    Parámetros:
    df : pd.DataFrame
        DataFrame de entrada.
    param_id_col : str
        Columna que no debe ser modificada.
    
    Retorna:
    pd.DataFrame
        DataFrame con las columnas de texto estandarizadas.
    """
    param_id_col= param_id_col['id_col']
    # Obtener las columnas de tipo object (string) excepto la columna id
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    if param_id_col in string_cols:
        string_cols.remove(param_id_col)

    # Diccionario de caracteres con tildes a reemplazar
    string_to_replace = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "ý": "y", "à": "a", "è": "e", "ì": "i", "ò": "o",
        "ù": "u", "ä": "a", "ë": "e", "ï": "i", "ö": "o",
        "ü": "u", "ÿ": "y", "â": "a", "ê": "e", "î": "i",
        "ô": "o", "û": "u", "ã": "a", "õ": "o", "@": "a",
        "ñ": "n"
    }
    
    # Caracteres especiales a reemplazar por "_"
    special_chars = r"[()/*\s:.\-;<>?/,'']"

    # Función que reemplaza caracteres en un string
    def replace_special_characters(text):
        if pd.isnull(text):
            return text
        # Reemplazar caracteres con tilde por los equivalentes sin tilde
        for key, value in string_to_replace.items():
            text = text.replace(key, value)
        # Reemplazar caracteres especiales por "_"
        text = re.sub(special_chars, "_", text)
        return text

    # Aplicar los reemplazos a cada columna de tipo string
    for col in string_cols:
        df[col] = df[col].apply(replace_special_characters)

    return df


#4. definir valores nulos fill nan 
def values_to_null(df: pd.DataFrame) -> pd.DataFrame:
    # (df:pd.DataFrame, param_buro_null: List[Any]) -> pd.DataFrame:
    """
    Reemplaza valores por nulos.
    """
    df = df.fillna(np.nan)

    return df

