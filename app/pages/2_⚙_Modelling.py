import streamlit as st
import pandas as pd
from io import StringIO

from app.core.system import AutoMLSystem

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


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

        # Read the dataset into a pd DataFrame
        try:
            # decode csv file and read using pd
            csv_str = selected_artifact.data.decode()
            df = pd.read_csv(StringIO(csv_str))
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.stop()

        st.write("### Dataset Preview")
        st.dataframe(df.head())

        # Downlad button for easier access to the dataset
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{selected_dataset_name}.csv",
            mime="text/csv",
        )

        all_columns = df.columns.tolist()

        # Selection of input features
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
            if target_dtype in ["int64", "float64"]:
                task_type = "Regression"
            elif target_dtype in ["object", "bool", "category"]:
                task_type = "Classification"

            st.write(f"Detected Task Type: {task_type}")
        else:
            st.info("Selecting a target feature is mandatory.")
else:
    st.info("There are no datasets available. Please upload one to proceed.")
