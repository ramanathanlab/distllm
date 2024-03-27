"""Search for text in a dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import ClassVar

import faiss
import numpy as np
import torch
from datasets import Dataset
from datasets.search import BatchedSearchResults

from distllm.embed import Encoder
from distllm.embed import Pooler


class FaissIndex:
    """FAISS index."""

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
            The path of the output dataset directory to write.
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
    ) -> BatchedSearchResults:
        """Search for the top k similar texts in the dataset.

        Parameters
        ----------
        query_embedding : np.ndarray | None
            The query embedding, by default None.
        top_k : int
            The number of top results to return, by default 1.

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


class Retriever:
    """Retriever for semantic similarity search."""

    def __init__(
        self,
        encoder: Encoder,
        pooler: Pooler,
        faiss_index: FaissIndex,
    ) -> None:
        """Initialize the Retriever.

        Parameters
        ----------
        encoder : Encoder
            The encoder instance to use for embedding queries.
        pooler : Pooler
            The pooler instance to use for pooling embeddings.
        faiss_index : FaissIndex
            The FAISS index instance to use for searching.
        """
        self.encoder = encoder
        self.pooler = pooler
        self.faiss_index = faiss_index

    def search(
        self,
        query: str | list[str] | None = None,
        query_embedding: np.ndarray | None = None,
        top_k: int = 1,
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
        results = self.faiss_index.search(query_embedding, top_k)

        return results, query_embedding

    @torch.no_grad()
    def get_pooled_embeddings(self, query: str | list[str]) -> np.ndarray:
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

        # Transform the embeddings according to the faiss strategy
        pool_embeds = self.faiss_index.transform(pool_embeds)

        return pool_embeds

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
        return np.array(self.faiss_index.get(indices, 'embeddings'))

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
        return self.faiss_index.get(indices, 'text')
