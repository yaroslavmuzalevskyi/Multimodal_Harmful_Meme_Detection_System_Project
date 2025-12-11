import os

from dotenv import load_dotenv
import torch


load_dotenv()


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", ""}


DATASET_NAME = "emily49/hateful-memes"

TEXT_MODEL_NAME = "distilbert-base-uncased"
VISION_MODEL_NAME = "google/vit-base-patch16-224-in21k"


MAX_TEXT_LENGTH = 64
IMAGE_SIZE = 224

BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1


DATASET_IMAGE_ROOT = os.getenv("DATASET_IMAGE_ROOT")
FIM_HIDDEN_DIM = int(os.getenv("FIM_HIDDEN_DIM", "256") or "256")
CLASSIFIER_HIDDEN_DIM = int(os.getenv("CLASSIFIER_HIDDEN_DIM", "256") or "256")
FREEZE_BASE_MODELS = _env_bool("FREEZE_BASE_MODELS", False)


SEED = 40

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = f"{CHECKPOINT_DIR}/best_model.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
