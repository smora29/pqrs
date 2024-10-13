"""
This is a boilerplate pipeline 'feature'
generated using Kedro 0.18.14
"""
from typing import Dict, List, Any, Tuple

import re
import pandas as pd
import numpy as np
import logging
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 0
def create_interactions(df):
    """Crea interacciones categóricas entre producto, problema y canal de envío."""
    
    # Limpiar los valores no válidos en 'Submitted via'
    valid_channels = ['web', 'phone', 'postal mail', 'fax', 'email', 'referral']
    df = df[df['submitted via'].isin(valid_channels)]
 
    #df['product_issue_interaction'] = df['product'] + "_" + df['issue']
    #df['subproduct_issue_interaction'] = df['sub-product'] + "_" + df['issue']
    #df['subproduct_via_interaction'] = df['sub-product'] + "_" + df['submitted via']
    #df['product_via_interaction'] = df['product'] + "_" + df['submitted via']
    return df


# 1. Vectorización de texto con TF-IDF
def vectorize_text_tfidf(df, columns, max_features=500):
    """Genera características TF-IDF para múltiples columnas específicas."""
    
    # Verificamos si 'columns' es una lista
    if not isinstance(columns, list):
        raise ValueError(f"El parámetro 'columns' debe ser una lista, pero se recibió: {type(columns)}")
    
    for column in columns:
        # Verificamos si la columna existe en el DataFrame
        if column not in df.columns:
            raise ValueError(f"La columna '{column}' no existe en el DataFrame")
        
        # Convertimos los valores de la columna a tipo string y manejamos los NaN
        df[column] = df[column].fillna('').astype(str)
        
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
        
        # Generamos la matriz TF-IDF para la columna
        tfidf_matrix = vectorizer.fit_transform(df[column])
        
        # Convertimos la matriz TF-IDF en un DataFrame
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        
        # Renombramos las columnas para identificar el origen
        tfidf_df.columns = [f'{column}_tfidf_{col}' for col in tfidf_df.columns]
        
        # Concatenar las nuevas columnas con el DataFrame original
        df = pd.concat([df, tfidf_df], axis=1)
    
    return df

# 2. Crear interacciones entre columnas categóricas


# 3. One-Hot Encoding para columnas categóricas
def encode_one_hot(df, columns):
    """Realiza One-Hot Encoding en múltiples columnas categóricas."""
    encoder = OneHotEncoder(sparse=False)
    df_encoded = df.copy()

    for column in columns:
        # Verificamos si la columna existe en el DataFrame
        if column not in df.columns:
            raise ValueError(f"La columna '{column}' no existe en el DataFrame")
        
        # Realizar el One-Hot Encoding en la columna actual
        encoded = encoder.fit_transform(df[[column]])
        
        # Crear DataFrame de las categorías codificadas
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))
        
        # Concatenar las nuevas columnas con el DataFrame original
        df_encoded = pd.concat([df_encoded.reset_index(drop=True), encoded_df], axis=1)
    
    return df_encoded


# 4. Calcular longitud del texto en una columna
def calculate_text_length(df, column):
    """Calcula la longitud de texto en una columna de texto."""
    df[column + '_text_length'] = df[column].apply(lambda x: len(str(x).split()))
    return df

# 5. Calcular sentimiento en una columna de texto
def calculate_sentiment(df, column):
    """Calcula el sentimiento de una columna de texto."""
    df[column + '_sentiment'] = df[column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return df

# 6 
def add_temporal_features(df):
    """
    Añade características temporales basadas en la fecha de recepción de la queja.
    
    Args:
    df (pd.DataFrame): DataFrame que contiene la columna 'date received'.
    
    Returns:
    pd.DataFrame: DataFrame con nuevas características temporales: 'date_received', 'day_of_week', y 'month'.
    """
    # Convertir la columna 'date received' a formato de fecha
    df['date_received'] = pd.to_datetime(df['date received'], format='%m_%d_%Y', errors='coerce')
    
    # Extraer el día de la semana (0: Lunes, 6: Domingo)
    df['day_of_week'] = df['date_received'].dt.dayofweek
    
    # Extraer el mes (1: Enero, 12: Diciembre)
    df['month'] = df['date_received'].dt.month
    
    return df

#7 


def calcular_importancia_caracteristicas(df, top_n):
    """
    Recibe un DataFrame, elimina columnas de tipo 'object' y entrena un modelo de Random Forest 
    para evaluar la importancia de las características. Retorna un DataFrame con la importancia de 
    las características y un nuevo DataFrame con las 'top_n' características más importantes.
    
    Parámetros:
    df (pd.DataFrame): DataFrame de entrada que contiene los datos, incluida la variable objetivo 'effective_response'.
    top_n (int): Número de características principales a seleccionar (por defecto 30).
    
    Retorna:
    importances_df (pd.DataFrame): DataFrame con las características y su importancia en el modelo.
    df_filtrado (pd.DataFrame): DataFrame con las 'top_n' características más importantes.
    """
    # Eliminar las columnas de tipo 'object' del DataFrame original
    df_numeric = df.select_dtypes(exclude=['object'])
    df_numeric = df_numeric.fillna(0)  # Reemplazar valores NaN con 0

    # Definir características (features) y variable objetivo (target)
    X = df_numeric.drop(columns=['effective_response', 'complaint id', 'date_received'])
    y = df_numeric['effective_response']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar un modelo de Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = rf_model.predict(X_test)

    # Obtener reporte de clasificación (no se usa, pero puedes agregarlo si lo necesitas)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)

    # Importancia de características
    feature_importances = rf_model.feature_importances_

    # Crear un DataFrame con las importancias
    importances_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Seleccionar las 'top_n' características más importantes
    top_features = importances_df.head(top_n)['Feature'].values

    # Crear un nuevo DataFrame filtrado con solo las 'top_n' características
    df_filtrado = df_numeric[top_features]
    df_filtrado['effective_response']=df_numeric['effective_response']
    df_filtrado['complaint id']=df_numeric['complaint id']
    df_filtrado['date_received']=df_numeric['date_received']
    
    
    return df_filtrado, importances_df
