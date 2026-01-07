"""Hybrid retrieval combining sparse and dense methods."""

from typing import List, Dict, Any, Tuple, Optional
from .base_retriever import BaseRetriever
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval system combining BM25 (sparse) and dense retrieval.

    Combines scores from both methods using a weighted combination,
    leveraging the strengths of both lexical matching and semantic similarity.
    """

    def __init__(
        self,
        corpus: List[Dict[str, Any]] = None,
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        alpha: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            corpus: List of documents to retrieve from
            dense_model_name: Name of the sentence transformer model
            alpha: Weight for dense retrieval (1-alpha for BM25). Range: [0, 1]
            device: Device to run model on
        """
        super().__init__(corpus)
        self.alpha = alpha
        self.bm25_retriever = BM25Retriever(corpus)
        self.dense_retriever = DenseRetriever(corpus, dense_model_name, device)

    def index(self, corpus: List[Dict[str, Any]]):
        """
        Index the corpus using both BM25 and dense retrieval.

        Args:
            corpus: List of documents with 'text' field
        """
        self.corpus = corpus
        print("Indexing with BM25...")
        self.bm25_retriever.index(corpus)
        print("Indexing with Dense Retriever...")
        self.dense_retriever.index(corpus)
        self.is_indexed = True

    def _normalize_scores(self, results: List[Tuple[Dict[str, Any], float]]) -> Dict[int, float]:
        """
        Normalize scores to [0, 1] range.

        Args:
            results: List of (document, score) tuples

        Returns:
            Dictionary mapping document index to normalized score
        """
        if not results:
            return {}

        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return {i: 1.0 for i in range(len(results))}

        normalized = {}
        for doc, score in results:
            doc_id = self.corpus.index(doc)
            normalized[doc_id] = (score - min_score) / (max_score - min_score)

        return normalized

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve top-k documents using hybrid scoring.

        Args:
            query: The search query
            top_k: Number of documents to retrieve

        Returns:
            List of (document, score) tuples
        """
        if not self.is_indexed:
            raise ValueError("Corpus not indexed. Call index() first.")

        # Retrieve from both systems (get more results for better fusion)
        retrieve_k = min(top_k * 3, len(self.corpus))

        bm25_results = self.bm25_retriever.retrieve(query, retrieve_k)
        dense_results = self.dense_retriever.retrieve(query, retrieve_k)

        # Normalize scores
        bm25_scores = self._normalize_scores(bm25_results)
        dense_scores = self._normalize_scores(dense_results)

        # Combine scores
        combined_scores = {}
        all_doc_ids = set(bm25_scores.keys()) | set(dense_scores.keys())

        for doc_id in all_doc_ids:
            bm25_score = bm25_scores.get(doc_id, 0.0)
            dense_score = dense_scores.get(doc_id, 0.0)
            combined_scores[doc_id] = (1 - self.alpha) * bm25_score + self.alpha * dense_score

        # Get top-k
        sorted_doc_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        results = []
        for doc_id, score in sorted_doc_ids:
            results.append((self.corpus[doc_id], score))

        return results
