import base64
from typing import Any, Dict, List, Optional


class Artifact:
    def __init__(
        self,
        name: str,
        asset_path: str,
        data: bytes,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
        type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Artifact class representing an asset stored in a specific location.
        """
        self._name = name
        self._asset_path = asset_path
        self._data = data
        self._version = version
        self._metadata = metadata if metadata is not None else {}
        self._type = type
        self._tags = tags if tags is not None else []
        self._id = (
            (
                f"{base64.b64encode(self.asset_path.encode()).decode()}"
                f":{self.version}"
            )
        )

    def read(self) -> bytes:
        """
        Reads the data from the artifact.
        """
        return self._data

    def save(self, new_data: bytes) -> None:
        """
        Saves new data to the artifact.
        """
        self._data = new_data

    def __repr__(self):
        return (
            f"Artifact(name=\"{self._name}\", type=\"{self._type}\", "
            f"id=\"{self._id}\")"
        )
