"""Base RAG strategy interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseRAGStrategy(ABC):
    """
    Abstract base class for RAG strategies.

    A RAG strategy defines how retrieved documents are incorporated
    into the LLM prompt to generate answers.
    """

    def __init__(self, max_context_length: int = 4000):
        """
        Initialize the RAG strategy.

        Args:
            max_context_length: Maximum tokens for context
        """
        self.max_context_length = max_context_length

    @abstractmethod
    def create_prompt(
        self,
        query: str,
        retrieved_docs: List[Tuple[Dict[str, Any], float]],
        **kwargs,
    ) -> str:
        """
        Create a prompt by incorporating retrieved documents.

        Args:
            query: The user's question
            retrieved_docs: List of (document, score) tuples
            **kwargs: Additional strategy-specific parameters

        Returns:
            The formatted prompt string
        """
        pass

    @abstractmethod
    def generate_answer(self, prompt: str, **kwargs) -> str:
        """
        Generate an answer using the LLM.

        Args:
            prompt: The formatted prompt
            **kwargs: Additional generation parameters

        Returns:
            The generated answer
        """
        pass

    def run(
        self,
        query: str,
        retrieved_docs: List[Tuple[Dict[str, Any], float]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline.

        Args:
            query: The user's question
            retrieved_docs: List of (document, score) tuples
            **kwargs: Additional parameters

        Returns:
            Dictionary with 'answer', 'prompt', and other metadata
        """
        prompt = self.create_prompt(query, retrieved_docs, **kwargs)
        answer = self.generate_answer(prompt, **kwargs)

        return {
            'answer': answer,
            'prompt': prompt,
            'num_docs': len(retrieved_docs),
            'query': query,
        }
