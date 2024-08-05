"""Utilities for distllm."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Literal
from typing import TypeVar
from typing import Union

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel

PathLike = Union[str, Path]

T = TypeVar('T')


class BaseConfig(BaseModel):
    """An interface to add JSON/YAML serialization to Pydantic models."""

    # A name literal to correctly identify and construct nested models
    # which have many possible options.
    name: Literal[''] = ''

    def write_json(self, path: PathLike) -> None:
        """Write the model to a JSON file.

        Parameters
        ----------
        path : str
            The path to the JSON file.
        """
        with open(path, 'w') as fp:
            json.dump(self.model_dump(), fp, indent=2)

    @classmethod
    def from_json(cls: type[T], path: PathLike) -> T:
        """Load the model from a JSON file.

        Parameters
        ----------
        path : str
            The path to the JSON file.

        Returns
        -------
        T
            A specific BaseConfig instance.
        """
        with open(path) as fp:
            data = json.load(fp)
        return cls(**data)

    def write_yaml(self, path: PathLike) -> None:
        """Write the model to a YAML file.

        Parameters
        ----------
        path : str
            The path to the YAML file.
        """
        with open(path, 'w') as fp:
            yaml.dump(
                json.loads(self.model_dump_json()),
                fp,
                indent=4,
                sort_keys=False,
            )

    @classmethod
    def from_yaml(cls: type[T], path: PathLike) -> T:
        """Load the model from a YAML file.

        Parameters
        ----------
        path : PathLike
            The path to the YAML file.

        Returns
        -------
        T
            A specific BaseConfig instance.
        """
        with open(path) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


def batch_data(data: list[T], chunk_size: int) -> list[list[T]]:
    """Batch data into chunks of size chunk_size.

    Parameters
    ----------
    data : list[T]
        The data to batch.
    chunk_size : int
        The size of each batch.

    Returns
    -------
    list[list[T]]
        The batched data.
    """
    batches = [
        data[i * chunk_size : (i + 1) * chunk_size]
        for i in range(0, len(data) // chunk_size)
    ]
    if len(data) > chunk_size * len(batches):
        batches.append(data[len(batches) * chunk_size :])
    return batches


def curl_download(data_file: Path, download_url: str) -> None:
    """Download a file using curl.

    Parameters
    ----------
    data_file : Path
        The path to the data file.

    download_url : str
        The URL to download the data from.
    """
    if not data_file.exists():
        command = f'curl -o {data_file} {download_url}'
        subprocess.run(command.split(), check=False)
