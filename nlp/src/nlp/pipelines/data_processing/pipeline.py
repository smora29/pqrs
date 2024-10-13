"""
Pipeline de la capa raw
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    #validar_columnas,
    convertir_a_minusculas,
    standardize_strings,
    values_to_null,
    #change_dtypes,
    #validate_unique_id_period_pd,
    #create_targets_pd
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=convertir_a_minusculas,
                inputs=["pqrs",
                        "parameters"],
                outputs="master_minusculas",
                name="raw_minusculas_columnas_node",
            ),
            node(
                func=standardize_strings,
                inputs=["master_minusculas",
                        "parameters"],
                outputs="master_standard",
                name="raw_standardize_strings_node",
            ),
            node(
                func=values_to_null,
                inputs=["master_standard"],
                outputs="master_null",
                name="raw_values_to_nulls_node",
            ),
            
        ]
    )