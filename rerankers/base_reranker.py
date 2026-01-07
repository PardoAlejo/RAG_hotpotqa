"""Base reranker interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseReranker(ABC):
    """
    Abstract base class for all re-ranking systems.

    A reranker takes an initial set of retrieved documents and reorders
    them to improve relevance ranking.
    """

    def __init__(self):
        """Initialize the reranker."""
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Tuple[Dict[str, Any], float]],
        top_k: int = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank documents based on their relevance to the query.

        Args:
            query: The search query
            documents: List of (document, initial_score) tuples
            top_k: Number of documents to return (None returns all)

        Returns:
            Reranked list of (document, new_score) tuples
        """
        pass

    def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[Tuple[Dict[str, Any], float]]],
        top_k: int = None,
    ) -> List[List[Tuple[Dict[str, Any], float]]]:
        """
        Rerank documents for multiple queries.

        Args:
            queries: List of search queries
            documents_list: List of document lists, one per query
            top_k: Number of documents to return per query

        Returns:
            List of reranked results, one per query
        """
        return [
            self.rerank(query, docs, top_k)
            for query, docs in zip(queries, documents_list)
        ]
