"""
This is a boilerplate pipeline 'model_output'
generated using Kedro 0.18.14
"""
from kedro.pipeline import Pipeline, node
from .nodes import model_output

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea y retorna el pipeline del modelo de salida.
    """
    return Pipeline(
        [
            node(
                func=model_output,
                inputs=["df", "importances_df", "params:params_model_output", "model_search"],
                outputs="df_calificado",
                name="model_output_node",
            )
        ]
    )
