"""
This is a boilerplate pipeline 'feature'
generated using Kedro 0.18.14
"""
from kedro.pipeline import Pipeline, node

from .nodes import (
    vectorize_text_tfidf,
    create_interactions,
    encode_one_hot, 
    calculate_text_length,
    calculate_sentiment,
    add_temporal_features,
    calcular_importancia_caracteristicas,
)


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            create_interactions,
            inputs="master_with_target",
            outputs="df_interactions",
            name="rcreate_interactions_node",
        ),
        node(
            vectorize_text_tfidf,
            inputs=["df_interactions", "params:tfidf"],
            outputs="df_tfidf_issue",
            name="vectorize_text_tfidf_node",
        ),        
        node(
            encode_one_hot,
            inputs=["df_tfidf_issue", "params:column_product"],
            outputs="df_encoded_product",
            name="create_interactions_node",
        ),
        node(
            calculate_text_length,
            inputs=["df_encoded_product", "params:column_issue"],
            outputs="df_with_text_length",
            name="calculate_text_length_node",
        ),
        node(
            calculate_sentiment,
            inputs=["df_with_text_length", "params:column_response"],
            outputs="df_with_sentiment",
            name="calculate_sentiment_node",
        ),
        node(
            add_temporal_features,
            inputs=["df_with_sentiment"],
            outputs="df_temp",
            name="add_temporal_features_node",
        ),
        node(
            calcular_importancia_caracteristicas,
            inputs=["df_temp","params:top_n"],
            outputs=["df_filtrado","importances_df"],
            name="importances_df_node",
        ),
    ])
