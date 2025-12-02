
import io
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoImageProcessor, get_linear_schedule_with_warmup

from config import (
    DATASET_NAME,
    TEXT_MODEL_NAME,
    VISION_MODEL_NAME,
    MAX_TEXT_LENGTH,
    BATCH_SIZE,
    SEED,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    CHECKPOINT_DIR,
    BEST_MODEL_PATH,
    DEVICE,
    DATASET_IMAGE_ROOT,
)
from model import MultimodalHatefulMemeModel


def get_tokenizer_and_image_processor():

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    image_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_NAME)
    return tokenizer, image_processor


def load_hateful_memes() -> Tuple[DatasetDict, Path | None]:

    print("Loading dataset metadata from Hugging Face cache...")
    dataset = load_dataset(DATASET_NAME)
    dataset_root = None
    if DATASET_IMAGE_ROOT:
        dataset_root = Path(DATASET_IMAGE_ROOT).expanduser().resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"DATASET_IMAGE_ROOT='{dataset_root}' does not exist. "
                "Set the env var to the directory containing the 'img/' files."
            )
    else:
        train_cache_files = dataset["train"].cache_files if "train" in dataset else []
        if train_cache_files:
            dataset_root = Path(train_cache_files[0]["filename"]).parent
    resolved_path = dataset_root if dataset_root is not None else "unknown"
    print(f"Dataset loaded. Image root resolved to: {resolved_path}")

    return dataset, dataset_root


def preprocess_function_builder(tokenizer, image_processor, dataset_root: Path | None):
    def preprocess(batch: Dict):
        texts: List[str] = batch["text"]
        text_enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
        )

        pil_images: List[Image.Image] = []
        for item in batch["img"]:
            img = None
            if isinstance(item, Image.Image):
                img = item
            elif isinstance(item, dict):
                if item.get("bytes") is not None:
                    img = Image.open(io.BytesIO(item["bytes"]))
                elif item.get("path") is not None:
                    img_path = Path(item["path"])
                    if not img_path.is_absolute() and dataset_root is not None:
                        # Try multiple possible locations
                        possible_paths = [
                            dataset_root / img_path,  # e.g., project_root/img/42953.png
                            dataset_root / img_path.name,  # e.g., project_root/42953.png
                            dataset_root / "img" / img_path.name,  # e.g., project_root/img/42953.png
                        ]
                        for possible_path in possible_paths:
                            if possible_path.exists():
                                img_path = possible_path
                                break
                    img = Image.open(img_path)
            else:
                # item is a string path
                img_path = Path(item)
                if not img_path.is_absolute() and dataset_root is not None:
                    # Try multiple possible locations
                    possible_paths = [
                        dataset_root / img_path,
                        dataset_root / img_path.name,
                        dataset_root / "img" / img_path.name,
                    ]
                    for possible_path in possible_paths:
                        if possible_path.exists():
                            img_path = possible_path
                            break
                img = Image.open(img_path)

            if img is None:
                raise FileNotFoundError(
                    f"Unable to load image for sample. "
                    f"Tried to load from: {img_path if 'img_path' in locals() else item}"
                )
            pil_images.append(img.convert("RGB"))

        img_enc = image_processor(
            pil_images,
            return_tensors="pt",
        )
        pixel_values = img_enc["pixel_values"]

        batch_size = len(texts)
        batch["pixel_values"] = [pixel_values[i] for i in range(batch_size)]

        batch["input_ids"] = text_enc["input_ids"]
        batch["attention_mask"] = text_enc["attention_mask"]

        return batch

    return preprocess


def prepare_dataloaders():

    torch.manual_seed(SEED)

    dataset, dataset_root = load_hateful_memes()

    print("Initializing tokenizer and image processor...")
    tokenizer, image_processor = get_tokenizer_and_image_processor()
    print("Tokenizer and image processor ready.")

    preprocess_fn = preprocess_function_builder(tokenizer, image_processor, dataset_root)

    print("Starting dataset preprocessing (tokenizing text + processing images)...")
    dataset = dataset.map(preprocess_fn, batched=True)
    print("Dataset preprocessing finished.")

    print("Building dataloaders...")
