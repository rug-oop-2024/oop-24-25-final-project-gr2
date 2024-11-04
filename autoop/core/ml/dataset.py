from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    def __init__(self, *args, **kwargs):
        """Dataset class to handle dataset artifacts."""
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0"
    ):
        """Create a Dataset object from a pandas DataFrame."""
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """Read the dataset from the artifact."""
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """Save the dataset to the artifact."""
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
