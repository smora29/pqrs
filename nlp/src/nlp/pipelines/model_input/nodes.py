"""
This is a boilerplate pipeline 'model input node'
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

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 1. Aplicar undersampling
def apply_undersampling(df: pd.DataFrame, target_reduction: float, random_state: int) -> pd.DataFrame:
    """
    Realiza undersampling para balancear las clases en el DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame de entrada que contiene la variable objetivo 'effective_response'.
        target_reduction (float): Proporción de reducción para la clase mayoritaria.
        random_state (int): Semilla aleatoria para reproducibilidad.

    Returns:
        pd.DataFrame: Un DataFrame balanceado después del undersampling.
    """
    class_counts = df["effective_response"].value_counts()
    majority_count_original = class_counts.max()
    majority_count_new = int(majority_count_original * (1 - target_reduction))

    sampling_strategy = {}
    for class_label, count in class_counts.items():
        sampling_strategy[class_label] = min(count, majority_count_new)

    under_sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = under_sampler.fit_resample(df.drop("effective_response", axis=1), df["effective_response"])

    df_resampled = pd.DataFrame(X_resampled, columns=df.drop("effective_response", axis=1).columns)
    df_resampled["effective_response"] = y_resampled
    logger.info("Distribución después del undersampling:")
    logger.info(df_resampled["effective_response"].value_counts())
    return df_resampled

# 2. Dividir los datos
def split_data(df: pd.DataFrame, params ) -> tuple:
    """
    Divide los datos en conjuntos de entrenamiento y prueba.

    Args:
        df (pd.DataFrame): El DataFrame balanceado que contiene las características y la variable objetivo.
        params (dict): Diccionario de parámetros, que contiene las columnas a eliminar y el nombre de la variable objetivo.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    drop_columns = params['drop_columns']
    target = params['target']
    test_size = params['test_size']
    random_state = params['random_state']
    
    df_1 = df.drop(drop_columns, axis=1)
    
    X = df_1.drop(target, axis=1)
    y = df_1[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    return X_train, X_test, y_train, y_test

# 3. Crear pipeline con ingeniería de características y XGBoost
def create_pipeline_with_feature_engineering(df: pd.DataFrame, scale_pos_weight: float) -> imbPipeline:
    """
    Crea un pipeline con preprocesamiento y el clasificador XGBoost.

    Args:
        df (pd.DataFrame): El DataFrame de entrada, ya balanceado y sin columnas irrelevantes.
        scale_pos_weight (float): El peso de la clase positiva para balancear el modelo.

    Returns:
        imbPipeline: Un pipeline que contiene el preprocesador y el clasificador.
    """
    # Eliminar las columnas innecesarias pero mantener la columna objetivo 'effective_response'
    df = df.drop(['complaint id', 'date_received', 'effective_response'], axis=1)

    # Detectar columnas numéricas y categóricas
    columnas_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    columnas_categoricas = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

    # Validar que existan columnas categóricas y numéricas, si no, ajustar el preprocesamiento
    transformers = []
    
    if len(columnas_numericas) > 0:
        transformers.append(("num", StandardScaler(), columnas_numericas))
    else:
        logger.warning("No hay columnas numéricas para escalar.")

    if len(columnas_categoricas) > 0:
        # Asegurarse de que las columnas categóricas están en formato adecuado
        for col in columnas_categoricas:
            if df[col].dtype != 'object':
                df[col] = df[col].astype('category')
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas))
    else:
        logger.warning("No hay columnas categóricas para codificar.")

    # Crear el preprocesador con las transformaciones necesarias
    preprocessor = ColumnTransformer(transformers)

    # Crear el pipeline con el preprocesador y el modelo XGBoost
    pipeline = imbPipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', scale_pos_weight=scale_pos_weight))
    ])

    return pipeline

# 4. Optimización de hiperparámetros usando RandomizedSearchCV
def model_selection_builder(pipeline: imbPipeline, params: dict) -> RandomizedSearchCV:
    """
    Realiza la optimización de hiperparámetros utilizando RandomizedSearchCV.

    Args:
        pipeline (imbPipeline): El pipeline de preprocesamiento y clasificación.
        params (dict): Diccionario con los parámetros para la búsqueda de hiperparámetros.

    Returns:
        RandomizedSearchCV: El modelo optimizado después de realizar la búsqueda.
    """
    param_grid = {
        "classifier__n_estimators": np.linspace(params["model_params"]["n_estimators"][0],
                                                params["model_params"]["n_estimators"][1],
                                                num=params["model_params"]["n_estimators"][2], dtype=int),
        "classifier__max_depth": np.linspace(params["model_params"]["max_depth"][0],
                                             params["model_params"]["max_depth"][1],
                                             num=params["model_params"]["max_depth"][2], dtype=int),
        "classifier__learning_rate": np.linspace(params["model_params"]["learning_rate"][0],
                                                 params["model_params"]["learning_rate"][1],
                                                 num=params["model_params"]["learning_rate"][2]),
        "classifier__subsample": np.linspace(params["model_params"]["subsample"][0],
                                             params["model_params"]["subsample"][1],
                                             num=params["model_params"]["subsample"][2]),
        "classifier__colsample_bytree": np.linspace(params["model_params"]["colsample_bytree"][0],
                                                    params["model_params"]["colsample_bytree"][1],
                                                    num=params["model_params"]["colsample_bytree"][2]),
        "classifier__gamma": np.linspace(params["model_params"]["gamma"][0],
                                         params["model_params"]["gamma"][1],
                                         num=params["model_params"]["gamma"][2]),
    }

    model_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=params["model_params"]["n_iter"],
        cv=params["model_params"]["cv"],
        scoring=params["model_params"]["scores"],
        random_state=params["random_state"],
        refit=params["model_params"]["refit"],
        verbose=params["model_params"]["verbose"],
    )

    return model_search

# 5. Crear el pipeline final, incluyendo balanceo de clases con SMOTE y ejecutar
def create_final_pipeline(df: pd.DataFrame, params: dict) -> RandomizedSearchCV:
    """
    Crea el pipeline final, ejecuta el balanceo, la división de datos y el entrenamiento del modelo.

    Args:
        df (pd.DataFrame): El DataFrame inicial.
        params (dict): Diccionario con los parámetros para el pipeline.

    Returns:
        RandomizedSearchCV: El modelo entrenado con la mejor configuración de hiperparámetros.
    """
    # Aplicar undersampling a las clases mayoritarias
    df_balanced = apply_undersampling(df, params["target_reduction"], params["random_state"])
    
    # Dividir los datos en conjuntos de entrenamiento y prueba con estratificación
    X_train, X_test, y_train, y_test = split_data(df_balanced, params)
    
    # Aplicar SMOTE para balancear el conjunto de entrenamiento
    smote = SMOTE(random_state=params["random_state"])
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Calcular scale_pos_weight dinámicamente
    class_counts = y_train_smote.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]
    logger.info(f"Scale_pos_weight calculado: {scale_pos_weight}")
    
    # Crear el pipeline con ingeniería de características y el clasificador, incluyendo el scale_pos_weight calculado
    pipeline = create_pipeline_with_feature_engineering(df_balanced, scale_pos_weight)
    
    # Realizar la búsqueda de hiperparámetros
    model_search = model_selection_builder(pipeline, params)
    
    # Ajustar el modelo en los datos de entrenamiento balanceados
    model_search.fit(X_train_smote, y_train_smote)
    
    # Predicciones para el conjunto de prueba y el conjunto de entrenamiento
    y_pred_train = model_search.predict(X_train_smote)
    y_pred_test = model_search.predict(X_test)
    
    # Reporte para el conjunto de entrenamiento
    report_train = classification_report(y_train_smote, y_pred_train)
    accuracy_train = accuracy_score(y_train_smote, y_pred_train)
    logger.info(f"Reporte de clasificación en el conjunto de entrenamiento:\n{report_train}")
    logger.info(f"Exactitud en el conjunto de entrenamiento: {accuracy_train}")
    
    # Reporte para el conjunto de prueba
    report_test = classification_report(y_test, y_pred_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    logger.info(f"Reporte de clasificación en el conjunto de prueba:\n{report_test}")
    logger.info(f"Exactitud en el conjunto de prueba: {accuracy_test}")
    
    # Devolver el modelo entrenado
    return model_search
