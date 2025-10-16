"""
Build text dataset for TRM training.
Creates a dataset from text files for next-token prediction.
"""

import json
import os

import numpy as np
from argdantic import ArgParser
from common import PuzzleDatasetMetadata
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

cli = ArgParser()


class DataProcessConfig(BaseModel):
    input_file: str = "dataset/input.txt"
    output_dir: str = "data/text"
    model_name: str = "TaylorAI/gte-tiny"
    seq_len: int = 256
    stride: int = 32  # Slide by 32 tokens for good coverage
    train_split: float = 0.95  # 95% train, 5% test


def tokenize_text(text: str, tokenizer, config: DataProcessConfig):
    """Tokenize text and create sliding windows for next-token prediction."""
    # Tokenize the entire text
    encoded = tokenizer(
        text,
        padding=False,
        truncation=False,
        return_tensors=None,
        add_special_tokens=True,
    )

    token_ids = encoded["input_ids"]
    print(f"Total tokens: {len(token_ids)}")

    # Create sliding windows
    inputs = []
    labels = []

    for i in range(0, len(token_ids) - config.seq_len, config.stride):
        window = token_ids[i : i + config.seq_len]
        if len(window) == config.seq_len:
            inputs.append(window)
            # Labels are shifted by 1 (next token prediction)
            labels.append(token_ids[i + 1 : i + config.seq_len + 1])

    print(
        f"Created {len(inputs)} examples with seq_len={config.seq_len}, stride={config.stride}"
    )

    return inputs, labels


def convert_subset(
    set_name: str,
    inputs_list: list,
    labels_list: list,
    config: DataProcessConfig,
    vocab_size: int,
    pad_id: int,
):
    """Convert tokenized text to dataset format."""

    # Generate dataset structure
    results = {
        k: []
        for k in [
            "inputs",
            "labels",
            "puzzle_identifiers",
            "puzzle_indices",
            "group_indices",
        ]
    }
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    # Each window is a separate "puzzle" and separate "group"
    for inp, out in tqdm(
        zip(inputs_list, labels_list),
        total=len(inputs_list),
        desc=f"Processing {set_name}",
    ):
        results["inputs"].append(np.array(inp, dtype=np.int32))
        results["labels"].append(np.array(out, dtype=np.int32))

        example_id += 1
        puzzle_id += 1

        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(0)

        # Each example is its own group for text training
        results["group_indices"].append(puzzle_id)

    # Convert to numpy arrays
    results = {
        "inputs": np.vstack(results["inputs"]),
        "labels": np.vstack(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=config.seq_len,
        vocab_size=vocab_size,
        pad_id=pad_id,
        ignore_label_id=pad_id,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["puzzle_indices"]) - 1,
        sets=["all"],
    )

    # Save metadata as JSON
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    print(f"{set_name} set: {len(results['inputs'])} examples")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    """Preprocess text file into dataset format."""

    # Check if input file exists
    if not os.path.exists(config.input_file):
        raise FileNotFoundError(
            f"Input file '{config.input_file}' not found. "
            f"Please create a text file at this location."
        )

    # Read input text
    print(f"Reading text from {config.input_file}")
    with open(config.input_file, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Text length: {len(text)} characters")

    # Load tokenizer
    print(f"Loading tokenizer from {config.model_name}")
    model = SentenceTransformer(config.model_name, trust_remote_code=True)
    tokenizer = model.tokenizer

    # Get vocab size and pad id
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id

    print(f"Vocab size: {vocab_size}")
    print(f"Pad token ID: {pad_id}")

    # Tokenize and create windows
    inputs, labels = tokenize_text(text, tokenizer, config)

    # Split into train/test
    split_idx = int(len(inputs) * config.train_split)
    train_inputs = inputs[:split_idx]
    train_labels = labels[:split_idx]
    test_inputs = inputs[split_idx:]
    test_labels = labels[split_idx:]

    print(f"\nTrain examples: {len(train_inputs)}")
    print(f"Test examples: {len(test_inputs)}")

    # Convert and save
    convert_subset("train", train_inputs, train_labels, config, vocab_size, pad_id)
    convert_subset("test", test_inputs, test_labels, config, vocab_size, pad_id)

    # Save identifiers
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    print(f"\nDataset saved to {config.output_dir}")


if __name__ == "__main__":
    cli()
