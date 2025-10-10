"""
Utilities for loading pretrained embeddings from sentence-transformers models.
"""

import torch
from sentence_transformers import SentenceTransformer


def load_pretrained_embeddings(model_name: str, expected_vocab_size: int, expected_embedding_dim: int):
    """
    Load embedding weights from a pretrained sentence-transformers model.

    Args:
        model_name: HuggingFace model name (e.g., 'Lajavaness/bilingual-embedding-small')
        expected_vocab_size: Expected vocabulary size (raise error if mismatch)
        expected_embedding_dim: Expected embedding dimension (raise error if mismatch)

    Returns:
        embedding_weight: torch.Tensor of shape [vocab_size, embedding_dim]
        tokenizer: The tokenizer from the pretrained model

    Raises:
        ValueError: If dimensions don't match expected values
    """
    # Load the pretrained model
    model = SentenceTransformer(model_name, trust_remote_code=True)

    # Extract word embeddings
    word_embeddings = model[0].auto_model.embeddings.word_embeddings
    embedding_weight = word_embeddings.weight.data.clone()

    # Get tokenizer
    tokenizer = model.tokenizer

    # Validate dimensions
    vocab_size, embedding_dim = embedding_weight.shape

    if vocab_size != expected_vocab_size:
        raise ValueError(
            f"Vocabulary size mismatch! "
            f"Pretrained model has {vocab_size} tokens, "
            f"but config specifies {expected_vocab_size}. "
            f"Please update vocab_size in config to {vocab_size}."
        )

    if embedding_dim != expected_embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch! "
            f"Pretrained model has {embedding_dim} dimensions, "
            f"but config specifies hidden_size={expected_embedding_dim}. "
            f"Please update hidden_size in config to {embedding_dim}."
        )

    return embedding_weight, tokenizer


def get_vocab_size_and_embedding_dim(model_name: str):
    """
    Get vocab size and embedding dim from a pretrained model without loading full weights.

    Args:
        model_name: HuggingFace model name

    Returns:
        vocab_size: int
        embedding_dim: int
        max_seq_length: int
    """
    model = SentenceTransformer(model_name, trust_remote_code=True)

    word_embeddings = model[0].auto_model.embeddings.word_embeddings
    vocab_size, embedding_dim = word_embeddings.weight.shape
    max_seq_length = model.max_seq_length

    return vocab_size, embedding_dim, max_seq_length
