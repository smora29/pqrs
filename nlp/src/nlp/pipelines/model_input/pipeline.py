"""
This is a boilerplate pipeline 'model_input'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    apply_undersampling,
    split_data,
    create_pipeline_with_feature_engineering,
    model_selection_builder,
    create_final_pipeline,
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=create_final_pipeline,
                inputs=["df_filtrado", "params"],
                outputs="trained_model",
                name="model_training_node",
            ),
        ]
    )