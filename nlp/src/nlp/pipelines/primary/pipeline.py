"""
This is a boilerplate pipeline 'primary'
generated using Kedro 0.18.14
"""

from kedro.pipeline import node, Pipeline , pipeline

from .nodes import (
    remove_duplicates,
    clean_dataframe_by_missing_values,
    impute_missing_values,
    create_effective_response,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= remove_duplicates,
                inputs=["master_null",],
                outputs="master_sin_duplicados",
                name="master_sin_duplicados_node",
            ),
            node(
                func=clean_dataframe_by_missing_values,
                inputs=["master_sin_duplicados",
                        "parameters"],
                outputs="master_drop",
                name="rclean_dataframe_by_missing_values_node",
            ),
            node(
                func=impute_missing_values,
                inputs=["master_drop"],
                outputs="master_new",
                name="impute_missing_values_node",
            ),
            node(
                func=create_effective_response,
                inputs=["master_new"],
                outputs="master_with_target",
                name="target_node",
            ),
        ]
    )