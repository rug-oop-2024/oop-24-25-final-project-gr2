from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """ Raised when a path is not found """
    def __init__(self, path):
        """ Initialize the exception """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """ Abstract class for storage """
    @abstractmethod
    def save(self, data: bytes, path: str):
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str):
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """ A simple local storage """
    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initialize the local storage with a given base path.

        Args:
            base_path (str, optional): The base path to use for storing data.
                Defaults to "./assets".
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data to a specified key.

        Args:
            data (bytes): The data to be saved.
            key (str): The key where the data will be stored.
            This is a relative path from the base path set for the storage.
        """
        path = self._join_path(key)
        # Ensure parent directories are created
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data from a specified key.

        Args:
            key (str): The key where the data is stored.
            This is a relative path from the base path set for the storage.

        Returns:
            bytes: The data read from the specified key.

        Raises:
            NotFoundError: If the specified key does not exist.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, "rb") as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete data at a specified key.

        Args:
            key (str): The key where the data is stored.
            This is a relative path from the base path set for the storage.
            Defaults to "/".

        Raises:
            NotFoundError: If the specified key does not exist.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        List all keys under a given prefix.

        Args:
            prefix (str): The prefix to list keys from.
                This is a relative path from the base path set for the storage.
                Defaults to "/".

        Returns:
            List[str]: A list of keys under the specified prefix.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        # Use os.path.join for compatibility across platforms
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [os.path.relpath(p, self._base_path)
                for p in keys if os.path.isfile(p)]

    def _assert_path_exists(self, path: str) -> None:
        """
        Asserts that a given path exists.
        If the path does not exist, it raises a NotFoundError.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """Join a given path with the base path of the storage.
        This method ensures paths are OS-agnostic.

        Args:
            path (str): The path to join with the base path.

        Returns:
            str: The joined path.
        """
        return os.path.normpath(os.path.join(self._base_path, path))
