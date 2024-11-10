from autoop.core.ml.artifact import Artifact
import pandas as pd
import io
import base64


class Dataset(Artifact):
    """ Dataset class to handle dataset artifacts. """

    def __init__(self, *args, **kwargs) -> None:
        """ Initialize a Dataset object. """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0"
    ) -> "Dataset":
        """Create a Dataset object from a pandas DataFrame."""
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """Read the dataset from the artifact."""
        artifact_dict = super().read()
        data_base64 = artifact_dict.get("data")
        if data_base64 is None:
            raise ValueError("Artifact does not contain data")

        bytes_data = base64.b64decode(data_base64)
        csv_str = bytes_data.decode()

        return pd.read_csv(io.StringIO(csv_str))

    def save(self, data: pd.DataFrame) -> "Artifact":
        """Save the dataset to the artifact."""
        bytes_data = data.to_csv(index=False).encode()
        return super().save(bytes_data)
