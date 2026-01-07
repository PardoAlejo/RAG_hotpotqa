"""BM25 sparse retrieval system."""

from typing import List, Dict, Any, Tuple
from .base_retriever import BaseRetriever


class BM25Retriever(BaseRetriever):
    """
    BM25 (Best Matching 25) sparse retrieval system.

    BM25 is a probabilistic ranking function used in information retrieval.
    It's based on the bag-of-words model and considers term frequency and
    inverse document frequency.

    Requires: rank-bm25 library
    """

    def __init__(self, corpus: List[Dict[str, Any]] = None, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.

        Args:
            corpus: List of documents to retrieve from
            k1: Controls term frequency saturation (default: 1.5)
            b: Controls document length normalization (default: 0.75)
        """
        super().__init__(corpus)
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.tokenized_corpus = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (can be improved with better tokenizers)."""
        return text.lower().split()

    def index(self, corpus: List[Dict[str, Any]]):
        """
        Index the corpus using BM25.

        Args:
            corpus: List of documents with 'text' field
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 library not found. Install with: pip install rank-bm25"
            )

        self.corpus = corpus
        self.tokenized_corpus = [self._tokenize(doc.get('text', '')) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        self.is_indexed = True

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve top-k documents using BM25 scoring.

        Args:
            query: The search query
            top_k: Number of documents to retrieve

        Returns:
            List of (document, score) tuples
        """
        if not self.is_indexed:
            raise ValueError("Corpus not indexed. Call index() first.")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append((self.corpus[idx], float(scores[idx])))

        return results
