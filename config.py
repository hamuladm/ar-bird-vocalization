import yaml
from dataclasses import dataclass, field
from pathlib import Path

_CFG_PATH = Path(__file__).parent / "config.yaml"


def _load_yaml(path=_CFG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


_raw = _load_yaml()

SAMPLE_RATE: int = _raw["audio"]["sample_rate"]
MAX_LENGTH: int = _raw["audio"]["max_length"]

SNAC_MODEL: str = _raw["snac"]["model"]
CODEBOOK_SIZE: int = _raw["snac"]["codebook_size"]
SNAC_N_LEVELS: int = _raw["snac"]["n_levels"]

_filt = _raw["filtering"]
MODEL_CHECKPOINT: str = _filt["model_checkpoint"]
N_FFT: int = _filt["n_fft"]
HOP_LENGTH: int = _filt["hop_length"]
N_MELS: int = _filt["n_mels"]
SPEC_MEAN: float = _filt["spec_mean"]
SPEC_STD: float = _filt["spec_std"]
MIN_SAMPLES_PER_CLASS: int = _filt["min_samples_per_class"]


@dataclass
class GatingConfig:
    confidence_threshold: float = _filt["confidence_threshold"]
    singularity_ratio: float = _filt["singularity_ratio"]
    require_label_match: bool = _filt["require_label_match"]


@dataclass
class PipelineConfig:
    gating: GatingConfig = field(default_factory=GatingConfig)
    model_checkpoint: str = MODEL_CHECKPOINT
    batch_size: int = _filt["batch_size"]
    device: str = "cuda"
    output_dir: str = _filt["output_dir"]
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS
    val_ratio: float = _raw["data"]["val_ratio"]
    test_ratio: float = _raw["data"]["test_ratio"]
    seed: int = _raw["data"]["seed"]


_data = _raw["data"]
TOKEN_DIR: Path = Path(_data["token_dir"])
VAL_RATIO: float = _data["val_ratio"]
TEST_RATIO: float = _data["test_ratio"]
SEED: int = _data["seed"]

_train = _raw["training"]
TRAIN_EPOCHS: int = _train["epochs"]
TRAIN_BATCH_SIZE: int = _train["batch_size"]
LEARNING_RATE: float = _train["learning_rate"]
WARMUP_STEPS: int = _train["warmup_steps"]
MAX_SEQ_LEN: int = _train["max_seq_len"]
NUM_WORKERS: int = _train["num_workers"]
SAVE_DIR: Path = Path(_train["save_dir"])

_model = _raw["model"]
N_EMBD: int = _model["n_embd"]
N_LAYER: int = _model["n_layer"]
N_HEAD: int = _model["n_head"]
N_POSITIONS: int = _model["n_positions"]

_wandb = _raw["wandb"]
WANDB_PROJECT: str = _wandb["project"]
WANDB_ENTITY: str = _wandb["entity"]