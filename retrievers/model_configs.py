"""Configuration and metadata for sentence transformer models."""

# Model configurations with metadata
AVAILABLE_MODELS = {
    "all-MiniLM-L6-v2": {
        "full_name": "sentence-transformers/all-MiniLM-L6-v2",
        "display_name": "MiniLM-L6 (Fast)",
        "size_mb": 80,
        "embedding_dim": 384,
        "speed": "⚡⚡⚡",  # Fast
        "quality": "⭐⭐⭐",  # Good
        "description": "Lightweight and fast, good balance for most tasks",
        "use_case": "Best for quick experiments and development"
    },
    "all-mpnet-base-v2": {
        "full_name": "sentence-transformers/all-mpnet-base-v2",
        "display_name": "MPNet-Base (Highest Quality)",
        "size_mb": 420,
        "embedding_dim": 768,
        "speed": "⚡",  # Slower (largest model)
        "quality": "⭐⭐⭐⭐⭐",  # Excellent
        "description": "Highest quality embeddings, largest model",
        "use_case": "Best overall accuracy for production"
    },
    "all-distilroberta-v1": {
        "full_name": "sentence-transformers/all-distilroberta-v1",
        "display_name": "DistilRoBERTa (Balanced)",
        "size_mb": 290,
        "embedding_dim": 768,
        "speed": "⚡⚡",  # Medium
        "quality": "⭐⭐⭐⭐",  # Very Good
        "description": "Good balance of quality and speed",
        "use_case": "Best for most use cases, good quality/speed trade-off"
    }
}

# Default model
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def get_model_info(model_key: str) -> dict:
    """
    Get information about a model.

    Args:
        model_key: Short model key (e.g., 'all-MiniLM-L6-v2')

    Returns:
        Dictionary with model metadata
    """
    return AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS[DEFAULT_MODEL])


def get_full_model_name(model_key: str) -> str:
    """
    Get the full HuggingFace model name.

    Args:
        model_key: Short model key

    Returns:
        Full model name for loading
    """
    return get_model_info(model_key)["full_name"]


def list_available_models() -> list:
    """Get list of available model keys."""
    return list(AVAILABLE_MODELS.keys())


def get_model_comparison_table() -> str:
    """Generate a comparison table of all models."""
    table = []
    table.append("| Model | Size | Dim | Speed | Quality | Best For |")
    table.append("|-------|------|-----|-------|---------|----------|")

    for key, info in AVAILABLE_MODELS.items():
        row = f"| {info['display_name']:20s} | {info['size_mb']:3d}MB | {info['embedding_dim']:3d} | {info['speed']:6s} | {info['quality']:8s} | {info['use_case']:30s} |"
        table.append(row)

    return "\n".join(table)
