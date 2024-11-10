import streamlit as st
import pandas as pd
from io import StringIO

from app.core.system import AutoMLSystem
from autoop.core.ml.feature import Feature
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import get_metric

from autoop.core.ml.model import (
    LassoRegressionModel,
    GBRModel,
    RandomForestRegressionModel,
    MultipleLinearRegression,
    GradientBoostingClassificationModel,
    LogisticRegressionModel,
    RandomForestClassificationModel,
)

# Available models
MODEL_OPTIONS = {
    "Classification": {
        "Logistic Regression": LogisticRegressionModel,
        "Random Forest Classifier": RandomForestClassificationModel,
        "Gradient Boosting Classifier": GradientBoostingClassificationModel,
    },
    "Regression": {
        "Lasso Regression": LassoRegressionModel,
        "Gradient Boosting Regressor": GBRModel,
        "Random Forest Regressor": RandomForestRegressionModel,
        "Multiple Linear Regression": MultipleLinearRegression,
    },
}

# Metrics
METRICS = {
    "Regression": [
        "mean_squared_error",
        "mean_absolute_error",
        "r_squared",
        "max_error",
    ],
    "Classification": ["accuracy",
                       "balanced_accuracy",
                       "recall",
                       "hamming_loss"],
}

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    """ Helper function to write text a specific color. """
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


def feature_type_to_str(f_dtype) -> str:
    """ Convert feature type to string. """
    if f_dtype in ["int64", "float64"]:
        return "numerical"
    elif f_dtype in ["object", "bool", "category"]:
        return "categorical"


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline to train "
    "a model on a dataset."
)

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

if datasets:
    # Create a dictionary to map dataset names to their Artifact objects
    dataset_dict = {ds.name: ds for ds in datasets}
    dataset_names = list(dataset_dict.keys())

    # Select box for choosing a dataset
    selected_dataset_name = st.selectbox(
        "Select a dataset 📊",
        options=dataset_names,
        help="Choose a dataset from the artifact registry to model.",
    )

    if selected_dataset_name:
        # Retrieve the selected Artifact
        selected_artifact = dataset_dict[selected_dataset_name]

        selected_dataset = Dataset(
            name=selected_artifact.name,
            asset_path=selected_artifact.asset_path,
            data=selected_artifact.data,
            version=selected_artifact.version,
            metadata=selected_artifact.metadata,
            tags=selected_artifact.tags
        )

        # Read the dataset into a pd DataFrame
        try:
            # decode csv file and read using pd
            csv_str = selected_artifact.data.decode()
            df = pd.read_csv(StringIO(csv_str))
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.stop()

        st.write("### Dataset Preview")
        st.dataframe(df)

        # Download button for easier access to the dataset
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{selected_dataset_name}.csv",
            mime="text/csv",
        )

        all_columns = df.columns.tolist()

        st.divider()
        st.subheader("Feature Selection")

        input_features = st.multiselect(
            "Select input features",
            options=all_columns,
            help="Select one or more input features for the model.",
        )
        if not input_features:
            st.warning("Select AT LEAST ONE input feature.")

        # Selection of target feature
        target_feature = st.selectbox(
            "Select target feature",
            options=[col for col in all_columns if col not in input_features],
            help="Choose ONE target feature for the model.",
        )

        if target_feature:
            # Determine the task type
            target_dtype = df[target_feature].dtype
            feature_type = feature_type_to_str(target_dtype)
            if feature_type == "numerical":
                task_type = "Regression"
            elif feature_type == "categorical":
                task_type = "Classification"

            st.write(f"**Detected Task Type**: {task_type}")
        else:
            st.info("Selecting a target feature is mandatory.")

        st.divider()
        st.subheader("Model Selection")

        # Selecting model
        available_models = MODEL_OPTIONS.get(task_type, {})
        model_names = list(available_models.keys())

        selected_model_name = st.selectbox(
            f"Select one of the following models for {task_type}",
            options=model_names,
            help="Choose a machine learning model.",
        )

        if selected_model_name:
            selected_model = available_models[selected_model_name]

        # Split ratio
        st.divider()
        st.subheader("Split Ratio")

        train_set_per = st.slider(
            "Select the percentage of data to be used for training",
            min_value=10,
            max_value=90,
            value=80,
            step=5,
            help="Define the split ratio of the data (training% | testing%).",
        )
        test_set_per = 100 - train_set_per
        st.write(f"Training Data: {train_set_per}%")
        st.write(f"Testing Data: {test_set_per}%")

        # Metrics selection

        st.divider()
        st.subheader("Metrics")

        if task_type == "Regression":
            available_metrics = METRICS["Regression"]
            select_metrics = st.multiselect(
                "Select metrics to evaluate the model",
                options=available_metrics,
                help="Choose one or more metrics to evaluate the model.",
            )
        elif task_type == "Classification":
            available_metrics = METRICS["Classification"]
            select_metrics = st.multiselect(
                "Select metrics to evaluate the model",
                options=available_metrics,
                help="Choose one or more metrics to evaluate the model.",
            )

        if not select_metrics:
            st.warning("Select AT LEAST ONE metric.")

        # Create the pipeline

        if st.button("Create Pipeline"):
            # Check that all required selections are made
            if not input_features:
                st.warning("Select AT LEAST ONE input feature.")
            elif not target_feature:
                st.warning("Selecting a target feature is mandatory.")
            elif not select_metrics:
                st.warning("Select AT LEAST ONE metric.")
            else:
                # Create Feature objects for input features
                input_feature_objs = []
                for feature_name in input_features:
                    feature_dtype = feature_type_to_str(df[feature_name].dtype)
                    input_feature_objs.append(
                        Feature(name=feature_name, feature_type=feature_dtype)
                    )

                target_feature_type = feature_type_to_str(
                    df[target_feature].dtype)
                target_feature_obj = Feature(
                    name=target_feature, feature_type=target_feature_type
                )

                model_obj = selected_model()

                metrics_objs = [
                    get_metric(metric) for metric in select_metrics
                ]

                pipeline = Pipeline(
                    metrics=metrics_objs,
                    dataset=selected_dataset,
                    model=model_obj,
                    input_features=input_feature_objs,
                    target_feature=target_feature_obj,
                    split=train_set_per / 100.0,
                )

                # Store the pipeline in session state
                st.session_state['pipeline'] = pipeline

                st.subheader("📑 Pipeline Summary")

                with st.expander("Click to view extended summary"):
                    st.markdown("### Model")
                    st.markdown(
                        f"<span style='color:green;'>**Chosen Model:**</span> {selected_model_name}<br>"
                        f"<span style='color:green;'>**Type:**</span> {pipeline.model.type.capitalize()}",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "<span style='color:green;'>**Parameters:**</span>",
                        unsafe_allow_html=True,
                    )
                    model_params = pipeline.model.parameters
                    if model_params:
                        st.json(model_params)
                    else:
                        st.write("No parameters set.")

                    st.divider()
                    st.markdown("### Input Features")
                    for feature in pipeline._input_features:
                        st.markdown(
                            f"<span style='color:green;'>- **{feature.name}**</span> ({feature.feature_type})",
                            unsafe_allow_html=True,
                        )

                    st.divider()
                    st.markdown("### Target Feature")
                    st.markdown(
                        f"<span style='color:green;'>- **{pipeline.target_feature.name}**</span> "
                        f"({pipeline.target_feature.feature_type})",
                        unsafe_allow_html=True,
                    )

                    st.divider()
                    st.markdown("### Split Ratio")
                    st.markdown(
                        f"<span style='color:green;'>- **Training Data:**</span> "
                        f"{train_set_per}%",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<span style='color:green;'>- **Testing Data:**</span> "
                        f"{test_set_per}%",
                        unsafe_allow_html=True,
                    )

                    st.divider()
                    st.markdown("### Metrics")
                    for metric in pipeline._metrics:
                        st.markdown(
                            f"<span style='color:purple;'>- **{metric.name.replace('_', ' ').capitalize()}**</span>",
                            unsafe_allow_html=True,
                        )
        # Check if a pipeline has been created
        if 'pipeline' in st.session_state:
            st.subheader("📈 Training and Evaluation")

            if st.button("Run Pipeline"):
                with st.spinner("Training the model..."):
                    results = st.session_state['pipeline'].execute()

                # Display training metrics
                st.markdown("### Training Metrics")
                train_metrics = results.get('train_metrics', [])
                for metric_obj, value in train_metrics:
                    st.write(f"**{metric_obj.name.replace('_', ' ').capitalize()}**: {value:.4f}")

                # Display testing metrics
                st.markdown("### Testing Metrics")
                test_metrics = results.get('test_metrics', [])
                for metric_obj, value in test_metrics:
                    st.write(f"**{metric_obj.name.replace('_', ' ').capitalize()}**: {value:.4f}")
        else:
            st.info("Create a pipeline first.")

else:
    st.info("There are no datasets available. Please upload one to proceed.")
