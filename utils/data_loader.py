"""Data loading and processing utilities for HotpotQA dataset."""

import json
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class HotpotQADataset:
    """
    HotpotQA dataset loader and processor.

    Dataset Structure:
    - _id: Unique identifier
    - question: The question text
    - answer: The correct answer (absent in test sets)
    - supporting_facts: List of [title, sent_id] indicating supporting sentences
    - context: List of [title, sentences] paragraphs
    - type: "comparison" or "bridge"
    - level: "easy", "medium", or "hard"
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the dataset loader.

        Args:
            data_dir: Directory containing the HotpotQA dataset files
        """
        self.data_dir = Path(data_dir)
        self.train_data = None
        self.dev_distractor_data = None
        self.dev_fullwiki_data = None

    def download_data(self):
        """
        Download HotpotQA dataset.

        Instructions:
        1. Visit: http://hotpotqa.github.io/
        2. Download the following files to data/raw/:
           - hotpot_train_v1.1.json
           - hotpot_dev_distractor_v1.json
           - hotpot_dev_fullwiki_v1.json

        Or use wget:
        wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
        wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
        wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json
        """
        print("Please download the dataset manually using the instructions in this method.")
        print(f"Save files to: {self.data_dir.absolute()}")

    def load_json(self, filepath: str) -> List[Dict]:
        """Load a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_train(self) -> List[Dict]:
        """Load training data."""
        if self.train_data is None:
            filepath = self.data_dir / "hotpot_train_v1.1.json"
            if not filepath.exists():
                raise FileNotFoundError(f"Training data not found at {filepath}")
            self.train_data = self.load_json(filepath)
        return self.train_data

    def load_dev_distractor(self) -> List[Dict]:
        """Load dev set with distractor setting (gold paragraphs + distractors)."""
        if self.dev_distractor_data is None:
            filepath = self.data_dir / "hotpot_dev_distractor_v1.json"
            if not filepath.exists():
                raise FileNotFoundError(f"Dev distractor data not found at {filepath}")
            self.dev_distractor_data = self.load_json(filepath)
        return self.dev_distractor_data

    def load_dev_fullwiki(self) -> List[Dict]:
        """Load dev set with fullwiki setting (requires retrieval)."""
        if self.dev_fullwiki_data is None:
            filepath = self.data_dir / "hotpot_dev_fullwiki_v1.json"
            if not filepath.exists():
                raise FileNotFoundError(f"Dev fullwiki data not found at {filepath}")
            self.dev_fullwiki_data = self.load_json(filepath)
        return self.dev_fullwiki_data

    def get_statistics(self, data: List[Dict]) -> Dict:
        """Get dataset statistics."""
        total = len(data)
        types = {}
        levels = {}

        for item in data:
            item_type = item.get('type', 'unknown')
            item_level = item.get('level', 'unknown')
            types[item_type] = types.get(item_type, 0) + 1
            levels[item_level] = levels.get(item_level, 0) + 1

        return {
            'total': total,
            'types': types,
            'levels': levels
        }

    def format_context(self, context: List[List]) -> str:
        """
        Format context paragraphs into a single string.

        Args:
            context: List of [title, sentences] pairs

        Returns:
            Formatted context string
        """
        formatted = []
        for title, sentences in context:
            formatted.append(f"Title: {title}")
            for sent in sentences:
                formatted.append(sent)
            formatted.append("")  # Empty line between paragraphs
        return "\n".join(formatted)

    def extract_supporting_context(self, item: Dict) -> Tuple[str, List[str]]:
        """
        Extract only the supporting facts from context.

        Args:
            item: A single data item

        Returns:
            Tuple of (formatted_context, supporting_sentences)
        """
        supporting_facts = item.get('supporting_facts', [])
        context = item.get('context', [])

        # Create a lookup for context
        context_dict = {title: sentences for title, sentences in context}

        supporting_sentences = []
        for title, sent_id in supporting_facts:
            if title in context_dict and sent_id < len(context_dict[title]):
                supporting_sentences.append(context_dict[title][sent_id])

        formatted = f"Question: {item['question']}\n\n"
        formatted += "Supporting Facts:\n"
        formatted += "\n".join(supporting_sentences)

        return formatted, supporting_sentences


def download_hotpotqa_data(save_dir: str = "data/raw"):
    """
    Helper function to download HotpotQA dataset using wget.

    Args:
        save_dir: Directory to save the downloaded files
    """
    os.makedirs(save_dir, exist_ok=True)

    urls = [
        "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
        "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
    ]

    print(f"Downloading HotpotQA dataset to {save_dir}...")
    print("Run the following commands:")
    print()
    for url in urls:
        filename = url.split('/')[-1]
        print(f"wget -P {save_dir} {url}")
    print()
    print("Or use curl:")
    for url in urls:
        filename = url.split('/')[-1]
        print(f"curl -o {save_dir}/{filename} {url}")
