"""Search for text in a dataset."""

from __future__ import annotations

import functools
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
from typing import ClassVar
from typing import Literal

import faiss
import numpy as np
import torch
from datasets import Dataset
from datasets.search import BatchedSearchResults
from pydantic import Field
from sentence_transformers.quantization import quantize_embeddings
from sentence_transformers.quantization import semantic_search_faiss
from tqdm import tqdm

from distllm.embed import Encoder
from distllm.embed import EncoderConfigs
from distllm.embed import get_encoder
from distllm.embed import get_pooler
from distllm.embed import Pooler
from distllm.embed import PoolerConfigs
from distllm.utils import BaseConfig
from distllm.utils import batch_data


def quantize_dataset(dataset_path: Path, precision: str) -> np.ndarray:
    """Quantize the embeddings in the dataset to the specified precision.

    Parameters
    ----------
    dataset_path : Path
        The path to the dataset.
    precision : str
        The desired precision for the embeddings. Valid options are:
        "float32", "uint8", "int8", "ubinary", and "binary".
        But FAISS only supports "float32", "uint8", and "ubinary".
    """
    # Load the dataset
    dataset = Dataset.load_from_disk(str(dataset_path))
    dataset.set_format('numpy', columns=['embeddings'])

    # Load the pre-computed fp32 embeddings
    embeddings = dataset['embeddings']

    # Quantize the embeddings
    quantized_embeddings = quantize_embeddings(embeddings, precision=precision)

    return quantized_embeddings


class FaissIndexV2Config(BaseConfig):
    """Configuration for the FAISS index."""

    name: Literal['faiss_index_v2'] = 'faiss_index_v2'  # type: ignore[assignment]

    dataset_dir: Path = Field(
        ...,
        description='The path to the HF dataset directory containing the '
        'document text and fp32 embeddings.',
    )
    faiss_index_path: Path = Field(
        ...,
        description='The path to the FAISS index.',
    )
    dataset_chunk_paths: list[Path] | None = Field(
        default=None,
        description='The paths to the dataset chunks, each containing an '
        'HF dataset with the document text and fp32 embeddings, to be '
        'quantized and added to the FAISS index during creation.',
    )
    precision: str = Field(
        default='float32',
        description='The desired precision for the embeddings '
        '[float32, ubinary].',
    )
    search_algorithm: str = Field(
        default='exact',
        description='The desired search algorithm [exact, hnsw].',
    )
    rescore_multiplier: int = Field(
        default=2,
        description='Oversampling factor for rescoring.',
    )
    num_quantization_workers: int = Field(
        default=1,
        description='The number of quantization process workers.',
    )


class FaissIndexV2:
    """FAISS index using sentence transformers.

    Supported FAISS indexes:
        - IndexFlatIP
        - IndexHNSWFlat
        - IndexBinaryFlat
        - IndexBinaryHNSW

    Supported embedding precision:
        - float32
        - ubinary

    Supported search algorithms:
        - exact
        - hnsw

    If the FAISS index does not exist, it will be created and saved to disk.
    Supports parallel quantization of HF dataset chunks using a process pool.

    For more information, see:
    https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/embedding-quantization/semantic_search_faiss.py
    """

    def __init__(  # noqa: PLR0913
        self,
        dataset_dir: Path,
        faiss_index_path: Path,
        dataset_chunk_paths: list[Path] | None = None,
        precision: str = 'float32',
        search_algorithm: str = 'exact',
        rescore_multiplier: int = 2,
        num_quantization_workers: int = 1,
    ) -> None:
        """Initialize the FAISS index.

        Parameters
        ----------
        dataset_dir : Path
            The path to the HF dataset directory containing
            the document text and fp32 embeddings.
        faiss_index_path : Path
            The path to the FAISS index, if it does not exist,
            it will be created and saved to this path.
        dataset_chunk_paths : list[Path], optional
            The paths to the dataset chunks, each containing
            an HF dataset with the document text and fp32 embeddings,
            to be quantized and added to the FAISS index during creation.
            Each dataset chunk is quantized in parallel using a process
            pool, and the quantized embeddings are concatenated and added
            to the index, by default None.
        precision : str, optional
            The desired precision for the embeddings, by default 'float32'.
            Supported options are 'float32' and 'ubinary'. If 'ubinary' is
            chosen, the embeddings will be quantized to an unsigned binary
            format, which is more memory efficient than 'float32'.
        search_algorithm : str, optional
            Whether to use exact search or approximate FAISS search,
            by default 'exact'. Supported options are 'exact' and 'hnsw'.
        rescore_multiplier : int, optional
            Oversampling factor for rescoring. The code will now search
            `top_k * rescore_multiplier` samples and then rescore to only
            keep `top_k`, by default 2.
        num_quantization_workers : int, optional
            The number of quantization process workers, by default 1.
        """
        self.dataset_dir = dataset_dir
        self.faiss_index_path = faiss_index_path
        self.dataset_chunk_paths = dataset_chunk_paths
        self.precision = precision
        self.search_algorithm = search_algorithm
        self.rescore_multiplier = rescore_multiplier
        self.num_workers = num_quantization_workers

        # Validate the precision and search algorithm
        if self.precision not in ('float32', 'ubinary'):
            raise ValueError(
                f'Invalid precision {precision}. '
                'Options: ["float32" and "ubinary"]',
            )
        if self.search_algorithm not in ('exact', 'hnsw'):
            raise ValueError(
                f'Invalid search_algorithm {search_algorithm}. '
                'Options: ["exact" and "hnsw"]',
            )

        # Load the  from disk
        self.dataset = Dataset.load_from_disk(str(dataset_dir))

        # Initialize the FAISS index
        if self.faiss_index_path.exists():
            print(f'Loading FAISS index from {self.faiss_index_path}')
            self.faiss_index = self._load_index_from_disk()
        else:
            print(f'Creating FAISS index at {self.faiss_index_path}')
            self.faiss_index = self._create_index()

    def _load_index_from_disk(self) -> faiss.Index:
        """Load the FAISS index from disk."""
        if self.precision in ('float32', 'uint8'):
            return faiss.read_index(str(self.faiss_index_path))
        else:
            return faiss.read_index_binary(str(self.faiss_index_path))

    def _create_index(self) -> faiss.Index:
        # Define the worker function for quantization
        func = functools.partial(quantize_dataset, precision=self.precision)

        # Check if the dataset is chunked
        if self.dataset_chunk_paths is None:
            embeddings = quantize_dataset(self.dataset_dir, self.precision)

        else:
            # Quantize the embeddings in each dataset chunk in parallel
            quantized_embeddings = []
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                for x in tqdm(
                    executor.map(func, self.dataset_chunk_paths),
                    desc='Quantizing embeddings',
                ):
                    quantized_embeddings.append(x)

            # Concatenate the quantized embeddings
            embeddings = np.concatenate(quantized_embeddings)

        print(
            f'Creating {self.precision} FAISS index using '
            f'{self.search_algorithm} search with embeddings '
            f'shape: {embeddings.shape}',
        )

        # Build the FAISS index (logic borrowed from
        # sentence_transformers.quantization.semantic_search_faiss)
        if self.precision in ('float32', 'uint8'):
            if self.search_algorithm == 'exact':
                # Use the inner product similarity for float32
                index = faiss.IndexFlatIP(embeddings.shape[1])
            else:
                # Use the HNSW algorithm for approximate search
                index = faiss.IndexHNSWFlat(embeddings.shape[1], 16)

        elif self.precision == 'ubinary':
            if self.search_algorithm == 'exact':
                # Use exact search with the binary index
                index = faiss.IndexBinaryFlat(embeddings.shape[1] * 8)
            else:
                # Use the HNSW algorithm for approximate search
                index = faiss.IndexBinaryHNSW(embeddings.shape[1] * 8, 16)
        else:
            raise ValueError(f'Invalid precision {self.precision}')

        # Add the embeddings to the index
        index.add(embeddings)

        print('Writing the index to disk...')

        # Save the index to disk
        if self.precision in ('float32', 'uint8'):
            faiss.write_index(index, str(self.faiss_index_path))
        else:
            faiss.write_index_binary(index, str(self.faiss_index_path))

        return index

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform the embeddings according to the FAISS strategy.

        Parameters
        ----------
        embeddings : np.ndarray
            The embeddings to transform.

        Returns
        -------
        np.ndarray
            The transformed embeddings.
        """
        # Normalize the embeddings for inner product search
        faiss.normalize_L2(embeddings)

        return embeddings

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 1,
        score_threshold: float = 0.0,
    ) -> BatchedSearchResults:
        """Search for the top k similar texts in the dataset.

        Parameters
        ----------
        query_embedding : np.ndarray
            The query embeddings.
        top_k : int
            The number of top results to return, by default 1.
        score_threshold : float
            The score threshold to use for filtering out results,
            by default we keep everything 0.0.

        Returns
        -------
        BatchedSearchResults
            A namedtuple with list[list[float]] (.total_scores) of scores for
            each  of the top_k returned items and a list[list[int]]]
            (.total_indices) of indices for each of the top_k returned items
            for each query.
        """
        # TODO: Decide if we should normalize the query embeddings here.
        # Normalize the query embeddings
        # faiss.normalize_L2(query_embeddings)

        t_start = time.perf_counter()
        # Search the index for the top k similar results
        # The list of search results is in the format:
        # [[{"corpus_id": int, "score": float}, ...], ...]
        results, *_ = semantic_search_faiss(
            query_embedding,
            corpus_index=self.faiss_index,
            corpus_precision=self.precision,
            top_k=top_k,
            rescore=self.precision != 'float32',
            rescore_multiplier=self.rescore_multiplier,
            exact=self.search_algorithm == 'exact',
        )

        print(f'Search time: {time.perf_counter() - t_start:.6f} seconds')
        print(f'Retrieved {len(results)} results')

        # Convert the search results to a BatchedSearchResults object
        results = BatchedSearchResults(
            total_scores=[[r['score'] for r in res] for res in results],
            total_indices=[[r['corpus_id'] for r in res] for res in results],
        )

        # Filter out results with the score threshold
        results = self._filter_search_by_score(results, score_threshold)

        return results

    def _filter_search_by_score(
        self,
        results: BatchedSearchResults,
        score_threshold: float,
    ) -> BatchedSearchResults:
        """Filter out results with scores below the threshold.

        Parameters
        ----------
        results : BatchedSearchResults
            The search results to filter.
        score_threshold : float
            The retrieval score threshold to use.

        Returns
        -------
        BatchedSearchResults
            The filtered search results.
        """
        # If the threshold is 0.0, return the results as is
        if not score_threshold:
            return results

        # Filter out results with scores not satisfying the threshold
        new_total_indices, new_total_scores = [], []
        for indices, scores in zip(
            results.total_indices,
            results.total_scores,
        ):
            # Keep only the indices and scores satisfying the threshold
            new_indices, new_scores = [], []
            for index, score in zip(indices, scores):
                # Assumes inner product similarity
                if score >= score_threshold:
                    new_indices.append(index)
                    new_scores.append(score)

            # Append the filtered indices and scores
            new_total_indices.append(new_indices)
            new_total_scores.append(new_scores)

        return BatchedSearchResults(
            total_indices=new_total_indices,
            total_scores=new_total_scores,
        )

    def get(self, indices: list[int], key: str) -> list[Any]:
        """Get the values of a key from the dataset for the given indices.

        Parameters
        ----------
        indices : list[int]
            The list of indices to get.
        key : str
            The key to get from the dataset.

        Returns
        -------
        Dataset
            The dataset for the given indices.
        """
        return [self.dataset[i][key] for i in indices]


class FaissIndexV1Config(BaseConfig):
    """Configuration for the FAISS index."""

    name: Literal['faiss_index_v1'] = 'faiss_index_v1'  # type: ignore[assignment]

    dataset_dir: Path = Field(
        ...,
        description='The path to the HF dataset directory containing the '
        'document text and fp32 embeddings.',
    )
    faiss_index_path: Path | None = Field(
        default=None,
        description='The path to the FAISS index file, by default None, '
        'in which case the FAISS index file is assumed to be '
        'in the same directory as the dataset with a .index extension.',
    )
    faiss_index_name: str = Field(
        'embeddings',
        description=' The name of the dataset field containing the FAISS '
        'index, by default "embeddings". If metric is "inner_product", '
        'the embeddings should be pre-normalized (by setting '
        'normalize_embeddings=True when computing the embeddings).',
    )
    metric: str = Field(
        'inner_product',
        description='The metric to use for the FAISS index '
        '["l2", "inner_product"], by default "inner_product".',
    )


class FaissIndexV1:
    """FAISS index.

    For a more scalable implementation, consider using FaissIndexV2.
    """

    METRICS: ClassVar[dict[str, Any]] = {
        'l2': faiss.METRIC_L2,
        'inner_product': faiss.METRIC_INNER_PRODUCT,
    }

    def __init__(
        self,
        dataset_dir: Path,
        faiss_index_file: Path | None = None,
        faiss_index_name: str = 'embeddings',
        metric: str = 'inner_product',
    ) -> None:
        """Initialize the FAISS index.

        Parameters
        ----------
        dataset_dir : Path
            The path to the HF dataset directory containing the
            document text and fp32 embeddings.
        faiss_index_file : Path, optional
            The path to the FAISS index file, by default None,
            in which case the FAISS index file is assumed to be
            in the same directory as the dataset with a .index extension.
        faiss_index_name : str
            The name of the dataset field containing the FAISS index,
            by default 'embeddings'. If metric is 'inner_product',
            the embeddings should be pre-normalized (by setting
            normalize_embeddings=True when computing the embeddings).
        metric : str
            The metric to use for the FAISS index ['l2', 'inner_product'],
            by default 'inner_product'.
        """
        self.dataset_dir = dataset_dir
        self.faiss_index_file = faiss_index_file
        self.faiss_index_name = faiss_index_name
        self.metric = metric

        warnings.warn(
            'FaissIndexV1 is deprecated and will be removed '
            'in a future version. Consider using FaissIndexV2.',
            category=DeprecationWarning,
            stacklevel=2,
        )

        # By default, the FAISS index file has the same name as the dataset
        # and is saved with a .index extension in the directory containing
        # the dataset
        if self.faiss_index_file is None:
            self.faiss_index_file = dataset_dir.with_suffix('.index')

        # Load the dataset from disk
        self.dataset = Dataset.load_from_disk(self.dataset_dir)

        # Initialize the FAISS index
        if self.faiss_index_file.exists():
            # Load the FAISS index from disk
            self.dataset.load_faiss_index(
                index_name=self.faiss_index_name,
                file=self.faiss_index_file,
            )
        else:
            # Create a new FAISS index
            self._create_index()

    def _create_index(self) -> None:
        # Build the FAISS index
        self.dataset.add_faiss_index(
            column=self.faiss_index_name,
            index_name=self.faiss_index_name,
            faiss_verbose=True,
            metric_type=self.METRICS[self.metric],
        )

        # Save the dataset to disk
        self.dataset.save_faiss_index(
            self.faiss_index_name,
            self.dataset_dir.with_suffix('.index'),
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 1,
        score_threshold: float = 0.0,
    ) -> BatchedSearchResults:
        """Search for the top k similar texts in the dataset.

        Parameters
        ----------
        query_embedding : np.ndarray | None
            The query embedding, by default None.
        top_k : int
            The number of top results to return, by default 1.
        score_threshold : float
            The score threshold to use for filtering out results,
            by default we keep everything 0.0.

        Returns
        -------
        BatchedSearchResults
            A namedtuple with list[list[float]] (.total_scores) of scores for
            each  of the top_k returned items and a list[list[int]]]
            (.total_indices) of indices for each of the top_k returned items
            for each query.
        """
        # Search the dataset for the top k similar results
        results = self.dataset.search_batch(
            index_name=self.faiss_index_name,
            queries=query_embedding,
            k=top_k,
        )

        # Assure that the results are pure floats, ints and not numpy types
        results = BatchedSearchResults(
            total_scores=results.total_scores.tolist(),
            total_indices=results.total_indices.tolist(),
        )

        # Filter out results with the score threshold
        results = self._filter_search_by_score(results, score_threshold)

        return results

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform the embeddings according to the FAISS strategy.

        Parameters
        ----------
        embeddings : np.ndarray
            The embeddings to transform.

        Returns
        -------
        np.ndarray
            The transformed embeddings.
        """
        # Normalize the embeddings for inner product search
        if self.metric == 'inner_product':
            faiss.normalize_L2(embeddings)

        return embeddings

    def _compare_score(self, score: float, score_threshold: float) -> bool:
        """Compare the score to the threshold.

        Parameters
        ----------
        score : float
            The score to compare.
        score_threshold : float
            The score threshold to compare against.

        Returns
        -------
        bool
            Whether the score satisfies the threshold.
        """
        # Higher scores are better
        if self.metric == 'inner_product':
            return score >= score_threshold

        # Lower scores are better
        if self.metric == 'l2':
            return score <= score_threshold

        # Keep everything if the metric is unknown
        return True

    def _filter_search_by_score(
        self,
        results: BatchedSearchResults,
        score_threshold: float,
    ) -> BatchedSearchResults:
        """Filter out results with scores below the threshold.

        Parameters
        ----------
        results : BatchedSearchResults
            The search results to filter.
        score_threshold : float
            The retrieval score threshold to use.

        Returns
        -------
        BatchedSearchResults
            The filtered search results.
        """
        # If the threshold is 0.0, return the results as is
        if not score_threshold:
            return results

        # Filter out results with scores not satisfying the threshold
        new_total_indices, new_total_scores = [], []
        for indices, scores in zip(
            results.total_indices,
            results.total_scores,
        ):
            # Keep only the indices and scores satisfying the threshold
            new_indices, new_scores = [], []
            for index, score in zip(indices, scores):
                if self._compare_score(score, score_threshold):
                    new_indices.append(index)
                    new_scores.append(score)

            # Append the filtered indices and scores
            new_total_indices.append(new_indices)
            new_total_scores.append(new_scores)

        return BatchedSearchResults(
            total_indices=new_total_indices,
            total_scores=new_total_scores,
        )

    def get(self, indices: list[int], key: str) -> list[Any]:
        """Get the values of a key from the dataset for the given indices.

        Parameters
        ----------
        indices : list[int]
            The list of indices to get.
        key : str
            The key to get from the dataset.

        Returns
        -------
        Dataset
            The dataset for the given indices.
        """
        return [self.dataset[i][key] for i in indices]


class RetrieverConfig(BaseConfig):
    """Configuration for the retriever."""

    faiss_config: FaissIndexV1Config | FaissIndexV2Config = Field(
        ...,
        description='Settings for the faiss index',
    )
    encoder_config: EncoderConfigs = Field(
        ...,
        description='Settings for the encoder',
    )
    pooler_config: PoolerConfigs = Field(
        ...,
        description='Settings for the pooler',
    )
    batch_size: int = Field(
        4,
        description='Batch size for the embedder model',
    )

    def get_retriever(self) -> Retriever:
        """Get the retriever."""
        # Initialize the encoder
        encoder = get_encoder(self.encoder_config.model_dump())

        # Initialize the pooler
        pooler = get_pooler(self.pooler_config.model_dump())

        # TODO: Once FaissIndexV1 is removed, remove this if-else block
        # Initialize the faiss index
        faiss_kwargs = self.faiss_config.model_dump(exclude={'name'})
        if self.faiss_config.name == 'faiss_index_v1':
            faiss_index = FaissIndexV1(**faiss_kwargs)
        else:
            faiss_index = FaissIndexV2(**faiss_kwargs)  # type: ignore[assignment]

        retriever = Retriever(
            encoder=encoder,
            pooler=pooler,
            faiss_index=faiss_index,
            batch_size=self.batch_size,
        )

        return retriever


class Retriever:
    """Retriever for semantic similarity search."""

    def __init__(
        self,
        encoder: Encoder,
        pooler: Pooler,
        faiss_index: FaissIndexV1 | FaissIndexV2,
        batch_size: int = 4,
    ) -> None:
        """Initialize the Retriever.

        Parameters
        ----------
        encoder : Encoder
            The encoder instance to use for embedding queries.
        pooler : Pooler
            The pooler instance to use for pooling embeddings.
        faiss_index : FaissIndex | FaissIndexV2
            The FAISS index instance to use for searching.
        batch_size : int
            The batch size to use for encoding queries, by default 4.
        """
        self.encoder = encoder
        self.pooler = pooler
        self.faiss_index = faiss_index
        self.batch_size = batch_size

    def search(
        self,
        query: str | list[str] | None = None,
        query_embedding: np.ndarray | None = None,
        top_k: int = 1,
        score_threshold: float = 0.0,
    ) -> tuple[BatchedSearchResults, np.ndarray]:
        """Search for text similar to the queries.

        Parameters
        ----------
        query : str | list[str]
            The single query or list of queries.
        query_embedding : np.ndarray | None
            The query embedding, by default None.
        top_k : int
            The number of top results to return, by default 1.
        score_threshold : float
            The score threshold to use for filtering out results,
            by default we keep everything 0.0.

        Returns
        -------
        BatchedSearchResults
            A namedtuple with list[list[float]] (.total_scores) of scores for
            each  of the top_k returned items and a list[list[int]]]
            (.total_indices) of indices for each of the top_k returned items
            for each query sequence.
        np.ndarray
            The embeddings of the queries
            (shape: [num_queries, embedding_size])

        Raises
        ------
        ValueError
            If both query and query_embedding are None.
        """
        # Check whether arguments are valid
        if query is None and query_embedding is None:
            raise ValueError(
                'Provide at least one of query or query_embedding.',
            )

        # Embed the queries
        if query_embedding is None:
            assert query is not None
            query_embedding = self.get_pooled_embeddings(query)

        # Search the dataset for the top k similar results
        results = self.faiss_index.search(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        return results, query_embedding

    def get_pooled_embeddings(self, query: str | list[str]) -> np.ndarray:
        """Get the pooled embeddings for the queries.

        Parameters
        ----------
        query : str | list[str]
            The single query or list of queries.

        Returns
        -------
        np.ndarray
            The embeddings of the queries
            (shape: [num_queries, embedding_size])
        """
        # Convert the query to a list if it is a single string
        if isinstance(query, str):
            query = [query]

        # Sort the data by length
        indices = sorted(range(len(query)), key=lambda i: len(query[i]))
        sorted_query = [query[i] for i in indices]

        # Batch the queries
        query_batches = batch_data(sorted_query, chunk_size=self.batch_size)

        # Get the pooled embeddings for the queries
        pool_embeds = []
        for batch in query_batches:
            pool_embeds.append(self._get_pooled_embeddings(batch))

        # Combine the pooled embeddings
        pool_embeds = np.concatenate(pool_embeds, axis=0)

        # Reorder the embeddings to match the original order
        pool_embeds = pool_embeds[np.argsort(indices)]

        return pool_embeds

    @torch.no_grad()
    def _get_pooled_embeddings(self, query: str | list[str]) -> np.ndarray:
        """Get the embeddings for the queries.

        Parameters
        ----------
        query : str | list[str]
            The single query or list of queries.

        Returns
        -------
        np.ndarray
            The embeddings of the queries
            (shape: [num_queries, embedding_size])
        """
        # Convert the query to a list if it is a single string
        if isinstance(query, str):
            query = [query]

        # Tokenize the query sequences
        batch_encoding = self.encoder.tokenizer(
            query,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )

        # Move the batch encoding to the device
        inputs = batch_encoding.to(self.encoder.device)

        # Embed the queries
        query_embeddings = self.encoder.encode(inputs)

        # Compute average embeddings for the queries
        pool_embeds = self.pooler.pool(query_embeddings, inputs.attention_mask)

        # Convert the query embeddings to numpy float32 for FAISS
        pool_embeds = pool_embeds.cpu().numpy().astype(np.float32)

        # TODO: Consider moving this into faiss index internals
        # Transform the embeddings according to the faiss strategy
        pool_embeds = self.faiss_index.transform(pool_embeds)

        return pool_embeds

    def get(self, indices: list[int], key: str) -> list[Any]:
        """Get the values of a key from the dataset for the given indices.

        Parameters
        ----------
        indices : list[int]
            The list of indices to get.
        key : str
            The key to get from the dataset.

        Returns
        -------
        list[Any]
            The values for the given indices.
        """
        return self.faiss_index.get(indices, key)

    def get_embeddings(self, indices: list[int]) -> np.ndarray:
        """Get the embeddings for the given indices.

        Parameters
        ----------
        indices : list[int]
            The list of indices returned from the search.

        Returns
        -------
        np.ndarray
            Array of embeddings (shape: [num_indices, embed_size])
        """
        return np.array(self.get(indices, 'embeddings'))

    def get_texts(self, indices: list[int]) -> list[str]:
        """Get the texts for the given indices.

        Parameters
        ----------
        indices : list[int]
            The list of indices returned from the search.

        Returns
        -------
        list[str]
            List of texts for the given indices.
        """
        return self.get(indices, 'text')
