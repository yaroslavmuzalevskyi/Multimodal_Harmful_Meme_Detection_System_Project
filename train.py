
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

    print("Dataloaders ready.")
    return train_loader, val_loader, test_loader, num_labels


def move_batch_to_device(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

    return {
        key: value.to(DEVICE) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_one_epoch(
    model: MultimodalHatefulMemeModel,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    criterion,
) -> float:

    model.train()
    running_loss = 0.0

    for batch in dataloader:
        batch = move_batch_to_device(batch)
        labels = batch["label"].float().unsqueeze(-1)

        optimizer.zero_grad()
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
        )
        loss = criterion(logits, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

    return running_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(
    model: MultimodalHatefulMemeModel,
    dataloader: DataLoader,
    criterion,
) -> Tuple[float, float]:

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch)
        labels = batch["label"].float().unsqueeze(-1)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
        )
        loss = criterion(logits, labels)
        running_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        correct += (preds.squeeze(-1) == labels.long().squeeze(-1)).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(1, len(dataloader))
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def train():

    print("Preparing dataloaders...")
    train_loader, val_loader, _, _ = prepare_dataloaders()
    print("Dataloaders prepared. Initializing model...")

    model = MultimodalHatefulMemeModel().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Starting epoch {epoch}/{NUM_EPOCHS}...")
        train_loss = run_one_epoch(
            model, train_loader, optimizer, scheduler, criterion
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved new best model to {BEST_MODEL_PATH}")
        print(f"Epoch {epoch} completed.\n")


if __name__ == "__main__":
    print("Starting training pipeline...")
    train()
