"""Dense retrieval system using embeddings."""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from .base_retriever import BaseRetriever


class DenseRetriever(BaseRetriever):
    """
    Dense retrieval using neural embeddings.

    Uses pre-trained sentence transformers to encode queries and documents
    into dense vectors, then performs similarity search.

    Requires: sentence-transformers library
    """

    def __init__(
        self,
        corpus: List[Dict[str, Any]] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize dense retriever.

        Args:
            corpus: List of documents to retrieve from
            model_name: Name of the sentence transformer model
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        super().__init__(corpus)
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embeddings = None

    def _load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers not found. Install with: pip install sentence-transformers"
                )
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def index(self, corpus: List[Dict[str, Any]]):
        """
        Index the corpus by computing embeddings.

        Args:
            corpus: List of documents with 'text' field
        """
        self._load_model()
        self.corpus = corpus

        texts = [doc.get('text', '') for doc in corpus]
        print(f"Encoding {len(texts)} documents...")
        self.embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=True
        )
        self.is_indexed = True

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve top-k documents using cosine similarity.

        Args:
            query: The search query
            top_k: Number of documents to retrieve

        Returns:
            List of (document, score) tuples
        """
        if not self.is_indexed:
            raise ValueError("Corpus not indexed. Call index() first.")

        self._load_model()

        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]

        # Compute cosine similarity
        scores = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append((self.corpus[idx], float(scores[idx])))

        return results
