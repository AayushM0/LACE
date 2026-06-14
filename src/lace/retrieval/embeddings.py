"""Embedding generation for LACE memory system.

Uses sentence-transformers for fully local, zero-API-key embeddings.
Model is downloaded once and cached by the sentence-transformers library.

Default model: all-MiniLM-L6-v2
  - 384 dimensions
  - ~22MB download
  - Fast inference (~50ms per batch on CPU)
  - Good quality for semantic similarity tasks
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


# ── Model singleton ───────────────────────────────────────────────────────────
# Load the model once and reuse. Loading takes ~1s, inference is fast.

_model: "SentenceTransformer | None" = None
_model_lock = threading.Lock()
_current_model_name: str | None = None


def get_model(model_name: str = "all-MiniLM-L6-v2") -> "SentenceTransformer":
    """Return the embedding model, loading it if necessary."""
    global _model, _current_model_name

    with _model_lock:
        if _model is None or _current_model_name != model_name:
            import os
            import warnings
            warnings.filterwarnings("ignore")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["HF_HUB_OFFLINE"] = "1"          # ← ADD THIS LINE
            os.environ["TRANSFORMERS_OFFLINE"] = "1"     # ← AND THIS LINE

            # Suppress transformers logging
            try:
                import transformers
                transformers.logging.set_verbosity_error()
            except Exception:
                pass

            try:
                import huggingface_hub
                huggingface_hub.logging.set_verbosity_error()
            except Exception:
                pass

            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(model_name)
            _current_model_name = model_name

    return _model


def embed_text(text: str, model_name: str = "all-MiniLM-L6-v2") -> list[float]:
    """Embed a single string into a vector.

    Args:
        text: The text to embed.
        model_name: Which sentence-transformers model to use.

    Returns:
        A list of floats representing the embedding vector.
    """
    model = get_model(model_name)
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_batch(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed multiple strings efficiently in a single batch.

    Much faster than calling embed_text() in a loop.

    Args:
        texts: List of strings to embed.
        model_name: Which sentence-transformers model to use.
        batch_size: How many to encode at once.

    Returns:
        List of embedding vectors, in the same order as input.
    """
    if not texts:
        return []

    model = get_model(model_name)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return [e.tolist() for e in embeddings]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Since we normalize embeddings on creation, this is just a dot product.
    Returns a value between -1.0 and 1.0 (higher = more similar).
    """
    import numpy as np
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b))