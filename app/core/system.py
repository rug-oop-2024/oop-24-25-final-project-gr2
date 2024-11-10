from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry:
    """ Artifact registry is a class that manages the artifacts in the system.
    It is a singleton class."""
    def __init__(self, database: Database, storage: Storage) -> None:
        """ Constructor of the class."""
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        # save the artifact in the storage
        """
        Registers an artifact in the system.

        Args:
            artifact: The artifact to register.
        """
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        Lists all the artifacts in the system.

        Args:
            type: The type of artifact to list.
            If None, all artifacts are listed.

        Returns:
            A list of Artifact objects.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> "Artifact":
        """
        Retrieves an artifact from the registry.

        Args:
            artifact_id: The id of the artifact.

        Returns:
            The artifact object with the given id.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Deletes an artifact from the registry.

        Args:
            artifact_id: The id of the artifact to delete.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """ Represents the AutoML system. It is a singleton class."""
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """ Constructor of the class."""
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """
        Returns the singleton instance of the AutoMLSystem.

        If the instance does not exist, it initializes it with
        default storage and database configurations.

        Returns:
            AutoMLSystem: The singleton instance of the AutoMLSystem.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(LocalStorage("./assets/dbo"))
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> "ArtifactRegistry":
        """ Returns the artifact registry of the system."""
        return self._registry
