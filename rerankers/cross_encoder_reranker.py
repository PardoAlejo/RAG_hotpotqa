"""Cross-encoder based reranker."""

from typing import List, Dict, Any, Tuple, Optional
from .base_reranker import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker using transformer models.

    Cross-encoders process query-document pairs jointly, allowing for
    more accurate relevance scoring compared to bi-encoders, at the
    cost of higher computational requirements.

    Requires: sentence-transformers library
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: Name of the cross-encoder model
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.model = None

    def _load_model(self):
        """Load the cross-encoder model."""
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers not found. Install with: pip install sentence-transformers"
                )
            self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank(
        self,
        query: str,
        documents: List[Tuple[Dict[str, Any], float]],
        top_k: int = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank documents using cross-encoder scoring.

        Args:
            query: The search query
            documents: List of (document, initial_score) tuples
            top_k: Number of documents to return (None returns all)

        Returns:
            Reranked list of (document, new_score) tuples
        """
        if not documents:
            return []

        self._load_model()

        # Prepare query-document pairs
        docs = [doc for doc, _ in documents]
        texts = [doc.get('text', '') for doc in docs]
        pairs = [[query, text] for text in texts]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Combine documents with new scores
        reranked = [(doc, float(score)) for doc, score in zip(docs, scores)]

        # Sort by new scores
        reranked.sort(key=lambda x: x[1], reverse=True)

        # Return top-k if specified
        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked
