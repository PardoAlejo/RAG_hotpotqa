"""Simple concatenation strategy for RAG."""

from typing import List, Dict, Any, Tuple
from .base_strategy import BaseRAGStrategy


class SimpleConcatenationStrategy(BaseRAGStrategy):
    """
    Simple concatenation strategy.

    Concatenates all retrieved documents and adds them to the prompt
    before the question. This is the most straightforward approach.
    """

    def __init__(self, max_context_length: int = 4000, llm_client=None):
        """
        Initialize simple concatenation strategy.

        Args:
            max_context_length: Maximum tokens for context
            llm_client: LLM client for generation (implement as needed)
        """
        super().__init__(max_context_length)
        self.llm_client = llm_client

    def create_prompt(
        self,
        query: str,
        retrieved_docs: List[Tuple[Dict[str, Any], float]],
        **kwargs,
    ) -> str:
        """
        Create prompt by concatenating all documents.

        Args:
            query: The user's question
            retrieved_docs: List of (document, score) tuples

        Returns:
            The formatted prompt
        """
        context_parts = []

        for i, (doc, score) in enumerate(retrieved_docs):
            text = doc.get('text', '')
            context_parts.append(f"Document {i+1}:\n{text}\n")

        context = "\n".join(context_parts)

        prompt = f"""Answer the following question based on the provided documents.

Context:
{context}

Question: {query}

Answer:"""

        return prompt

    def generate_answer(self, prompt: str, **kwargs) -> str:
        """
        Generate answer using LLM.

        Args:
            prompt: The formatted prompt
            **kwargs: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            Generated answer
        """
        if self.llm_client is None:
            return "[LLM client not configured - implement your LLM interface]"

        # Implement your LLM call here
        # Example with OpenAI:
        # response = self.llm_client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=kwargs.get('temperature', 0.7),
        #     max_tokens=kwargs.get('max_tokens', 500)
        # )
        # return response.choices[0].message.content

        return "[Not implemented - configure LLM client]"
