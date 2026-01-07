"""Base retriever interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval systems.

    A retriever takes a query and returns relevant documents/passages
    from a corpus.
    """

    def __init__(self, corpus: List[Dict[str, Any]] = None):
        """
        Initialize the retriever.

        Args:
            corpus: List of documents to retrieve from
        """
        self.corpus = corpus or []
        self.is_indexed = False

    @abstractmethod
    def index(self, corpus: List[Dict[str, Any]]):
        """
        Index the corpus for efficient retrieval.

        Args:
            corpus: List of documents to index
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve top-k most relevant documents for the query.

        Args:
            query: The search query
            top_k: Number of documents to retrieve

        Returns:
            List of (document, score) tuples, sorted by relevance (highest first)
        """
        pass

    def batch_retrieve(
        self, queries: List[str], top_k: int = 5
    ) -> List[List[Tuple[Dict[str, Any], float]]]:
        """
        Retrieve documents for multiple queries.

        Args:
            queries: List of search queries
            top_k: Number of documents to retrieve per query

        Returns:
            List of retrieval results, one per query
        """
        return [self.retrieve(query, top_k) for query in queries]

    def get_corpus_size(self) -> int:
        """Return the size of the indexed corpus."""
        return len(self.corpus)
