import io
from typing import Dict, List

import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoImageProcessor

from config import (
    DATASET_NAME,
    TEXT_MODEL_NAME,
    VISION_MODEL_NAME,
    MAX_TEXT_LENGTH,
    BATCH_SIZE,
    SEED,
)


def get_tokenizer_and_image_processor():

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    image_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_NAME)
    return tokenizer, image_processor


def load_hateful_memes():

    dataset = load_dataset(DATASET_NAME)
    return dataset


def build_image_base_url() -> str:

    return f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/main/"


BASE_IMG_URL = build_image_base_url()


def fetch_image(path: str) -> Image.Image:

    url = BASE_IMG_URL + path
    resp = requests.get(url)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return img


def preprocess_function_builder(tokenizer, image_processor):

    def preprocess(batch: Dict):
        texts: List[str] = batch["text"]
        text_enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
        )


        img_paths: List[str] = batch["img"]

        images: List[Image.Image] = [fetch_image(p) for p in img_paths]

        img_enc = image_processor(
            images,
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

    dataset = load_hateful_memes()

    tokenizer, image_processor = get_tokenizer_and_image_processor()

    preprocess_fn = preprocess_function_builder(tokenizer, image_processor)

    dataset = dataset.map(preprocess_fn, batched=True)


    base_columns = ["input_ids", "attention_mask", "pixel_values", "label"]

    if "train" in dataset:
        dataset["train"].set_format(type="torch", columns=base_columns)

    val_split_name = "validation" if "validation" in dataset else "dev"
    if val_split_name in dataset:
        dataset[val_split_name].set_format(type="torch", columns=base_columns)

    test_loader = None
    if "test" in dataset:
        test_cols = [c for c in base_columns if c in dataset["test"].column_names]
        dataset["test"].set_format(type="torch", columns=test_cols)

    train_loader = DataLoader(
        dataset["train"], batch_size=BATCH_SIZE, shuffle=True
    )

    val_loader = DataLoader(
        dataset[val_split_name], batch_size=BATCH_SIZE, shuffle=False
    )

    if "test" in dataset:
        test_loader = DataLoader(
            dataset["test"], batch_size=BATCH_SIZE, shuffle=False
        )

    num_labels = 1

    return train_loader, val_loader, test_loader, num_labels
