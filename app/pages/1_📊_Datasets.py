import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact

automl = AutoMLSystem.get_instance()

st.title("Dataset Uploader")

st.write("Available datasets:")
datasets = automl.registry.list(type="dataset")

existing_names = [ds.name for ds in datasets]

if datasets:
    for ds in datasets:
        st.write(f"- {ds.name}")
else:
    st.write("No datasets available.")

# Upload CSV file
input_file = st.file_uploader("Choose a CSV file", type=["csv"])

if input_file is not None:
    try:
        data_frame = pd.read_csv(input_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    st.write("### Preview of the dataset:")
    st.dataframe(data_frame.head())

    dataset_name = st.text_input(
        "Enter a name for the dataset", value=input_file.name.split(".")[0]
    )

    if dataset_name in existing_names:
        st.warning("Name already exists. Please choose a different name.")
    else:
        if st.button("Save Dataset"):
            try:
                # Define the asset path with a .csv extension
                asset_path = f"datasets/{dataset_name}.csv"

                dataset = Dataset.from_dataframe(
                    data=data_frame,
                    name=dataset_name,
                    asset_path=asset_path
                )

                # encode to bytes
                csv_bytes = data_frame.to_csv(index=False).encode()

                # Create an Artifact with the csv dataset
                artifact = Artifact(
                    name=dataset_name,
                    asset_path=asset_path,
                    data=csv_bytes,
                    type="dataset",
                    metadata={"source": "uploaded_csv"},
                    tags=["csv", "dataset"],
                )

                automl.registry.register(artifact)

                st.success(
                    f"'{dataset_name}' has been uploaded and registered."
                )
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
