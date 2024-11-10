import streamlit as st
import pickle

from app.core.system import AutoMLSystem

st.set_page_config(page_title="Deployment", page_icon="📂")

st.write("# 📂 Deployment")

automl = AutoMLSystem.get_instance()

pipelines = automl.registry.list(type="pipeline")

if pipelines:
    pipeline_dict = {f"{p.name} (v{p.version})": p for p in pipelines}
    pipeline_names = list(pipeline_dict.keys())

    selected_pipeline_name = st.selectbox(
        "Select a pipeline to deploy",
        options=pipeline_names,
        help="Choose a saved pipeline from the artifact registry.",
    )

    if selected_pipeline_name:
        pipeline_artifact = pipeline_dict[selected_pipeline_name]

        try:
            pipeline = pickle.loads(pipeline_artifact.data)
        except Exception as e:
            st.error(f"Failed to load pipeline: {e}")
            st.stop()

        # selected pipeline summary
        st.subheader("📑 Pipeline Summary")
        st.write(f"**Name:** {pipeline_artifact.name}")
        st.write(f"**Version:** {pipeline_artifact.version}")

        task_type = pipeline_artifact.metadata.get('task_type', 'N/A')
        st.write(f"**Task Type:** {task_type}")

        model_name = pipeline_artifact.metadata.get('model_name', 'N/A')
        st.write(f"**Model:** {model_name}")

        input_features = ', '.join(
            pipeline_artifact.metadata.get('input_features', [])
        )
        st.write(f"**Input Features:** {input_features}")

        target_feature = pipeline_artifact.metadata.get(
            'target_feature', 'N/A'
        )
        st.write(f"**Target Feature:** {target_feature}")

        metrics = ', '.join(pipeline_artifact.metadata.get('metrics', []))
        st.write(f"**Metrics:** {metrics}")

        split_ratio = pipeline_artifact.metadata.get('split_ratio', 'N/A')
        st.write(f"**Split Ratio:** {split_ratio}")

        tags = ', '.join(pipeline_artifact.tags)
        st.write(f"**Tags:** {tags}")

        st.divider()
