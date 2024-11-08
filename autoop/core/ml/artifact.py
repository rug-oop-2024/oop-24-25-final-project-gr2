import base64
from typing import Any, Dict, List, Optional
from copy import deepcopy


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
        self._id = self._generate_id()

    def _generate_id(self) -> str:
        if not self._asset_path:
            raise ValueError("asset_path must be non-empty.")
        if not self._version:
            raise ValueError("version must be non-empty.")

        encoded_path = base64.urlsafe_b64encode(
            self._asset_path.encode()
        ).decode()
        return f"{encoded_path}:{self._version}"

    @property
    def id(self) -> str:
        return self._id

    @property
    def asset_path(self) -> str:
        return self._asset_path

    @property
    def version(self) -> str:
        return self._version

    @property
    def data(self) -> bytes:
        return self._data

    @property
    def name(self) -> str:
        return self._name

    @property
    def metadata(self) -> Dict[str, Any]:
        return deepcopy(self._metadata)

    @property
    def type(self) -> str:
        return self._type

    @property
    def tags(self) -> List[str]:
        return deepcopy(self._tags)

    def read(self) -> Dict[str, Any]:
        """
        Converts the artifact to a dictionary.
        """
        return {
            "name": self._name,
            "asset_path": self._asset_path,
            "version": self._version,
            "data": base64.b64encode(self._data).decode(),
            "metadata": self._metadata,
            "type": self._type,
            "tags": self._tags,
        }

    @classmethod
    def save(cls, data: Dict[str, Any]) -> "Artifact":
        """
        Makes an artifact from a dictionary.
        """
        return Artifact(
            name=data.get("name", ""),
            asset_path=data["asset_path"],
            version=data["version"],
            data=base64.b64decode(data["data"]),
            metadata=data.get("metadata", {}),
            type=data.get("type", "generic"),
            tags=data.get("tags", []),
        )

    def __str__(self):
        return f"Artifact(name={self._name}, asset_path={self._asset_path}, \
            version={self._version})"
