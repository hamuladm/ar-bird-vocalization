import yaml
from dataclasses import dataclass, field
from pathlib import Path

_CFG_PATH = Path(__file__).parent / "config.yaml"


def _load_yaml(path=_CFG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


_raw = _load_yaml()

DEVICE: str = _raw["device"]

SAMPLE_RATE: int = _raw["audio"]["sample_rate"]
MAX_LENGTH: int = _raw["audio"]["max_length"]

_snac = _raw["snac"]
SNAC_MODEL: str = _snac["model"]
CODEBOOK_SIZE: int = _snac["codebook_size"]
SNAC_N_LEVELS: int = _snac["n_levels"]

_snac_gen = _snac.get("generation", {})
SNAC_GEN_TEMPERATURE: float = _snac_gen.get("temperature", 1.0)
SNAC_GEN_TOP_K: int = _snac_gen.get("top_k", 50)

_snac_inf = _snac.get("inference", {})
SNAC_INF_BATCH_SIZE: int = _snac_inf.get("batch_size", 32)
SNAC_INF_NUM_WORKERS: int = _snac_inf.get("num_workers", 8)

_taxonomy = _raw["taxonomy"]
TAXONOMY_API_URL: str = _taxonomy["api_url"]
TAXONOMY_CACHE: str = _taxonomy["cache_path"]

_logging = _raw["logging"]
LOG_DIR: str = _logging["log_dir"]

_filt = _raw["filtering"]
FILTERED_DIR: str = _filt["output_dir"]
NUM_XCL_CLASSES: int = _filt["num_xcl_classes"]
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
    num_workers: int = _filt["num_workers"]
    device: str = DEVICE
    output_dir: str = _filt["output_dir"]
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS
    val_ratio: float = _raw["data"]["val_ratio"]
    test_ratio: float = _raw["data"]["test_ratio"]
    seed: int = _raw["data"]["seed"]

_relaxed = _raw["relaxed_filtering"]
RELAXED_FILTERED_DIR: str = _relaxed["output_dir"]


@dataclass
class RelaxedGatingConfig:
    top_k: int = _relaxed["top_k"]
    max_entropy: float = _relaxed["max_entropy"]


@dataclass
class RelaxedPipelineConfig:
    gating: RelaxedGatingConfig = field(default_factory=RelaxedGatingConfig)
    model_checkpoint: str = MODEL_CHECKPOINT
    batch_size: int = _relaxed["batch_size"]
    num_workers: int = _relaxed["num_workers"]
    device: str = DEVICE
    output_dir: str = _relaxed["output_dir"]
    min_samples_per_class: int = _relaxed["min_samples_per_class"]
    taxonomy_cache: str = TAXONOMY_CACHE
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
WANDB_PROJECT: str = _wandb["wandb_project"]
WANDB_ENTITY: str = _wandb["wandb_entity"]

_ag = _raw.get("audiogen", {})
AG_PRETRAINED: str = _ag.get("pretrained", "facebook/audiogen-medium")
AG_SAMPLE_RATE: int = _ag.get("sample_rate", 16000)
AG_TARGET_LENGTH: float = _ag.get("target_length", 10)
AG_SAVE_DIR: Path = Path(_ag.get("save_dir", "checkpoints/audiogen"))
AG_NUM_WORKERS: int = _ag.get("num_workers", 8)
AG_GRAD_ACCUM: int = _ag.get("grad_accum", 1)

_ag_tokens = _ag.get("encodec_tokens", {})
AG_RELAXED_TOKEN_DIR: str = _ag_tokens.get("relaxed_dir", "data/relaxed_encodec_tokens")
AG_STRICT_TOKEN_DIR: str = _ag_tokens.get("strict_dir", "data/strict_encodec_tokens")

_ag_enc_inf = _ag.get("encodec_inference", {})
AG_ENCODEC_BATCH_SIZE: int = _ag_enc_inf.get("batch_size", 32)
AG_ENCODEC_NUM_WORKERS: int = _ag_enc_inf.get("num_workers", 8)


@dataclass
class AudioGenStageConfig:
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 1000


_s1 = _ag.get("stage1", {})
AG_STAGE1 = AudioGenStageConfig(
    epochs=_s1.get("epochs", 20),
    batch_size=_s1.get("batch_size", 8),
    learning_rate=_s1.get("learning_rate", 1e-3),
    warmup_steps=_s1.get("warmup_steps", 100),
)

_s2 = _ag.get("stage2", {})
AG_STAGE2 = AudioGenStageConfig(
    epochs=_s2.get("epochs", 50),
    batch_size=_s2.get("batch_size", 4),
    learning_rate=_s2.get("learning_rate", 1e-4),
    warmup_steps=_s2.get("warmup_steps", 1000),
)

_s3 = _ag.get("stage3", {})
AG_STAGE3 = AudioGenStageConfig(
    epochs=_s3.get("epochs", 50),
    batch_size=_s3.get("batch_size", 4),
    learning_rate=_s3.get("learning_rate", 5e-5),
    warmup_steps=_s3.get("warmup_steps", 500),
)

_ag_gen = _ag.get("generation", {})
AG_GEN_DURATION: float = _ag_gen.get("duration", 5.0)
AG_GEN_TEMPERATURE: float = _ag_gen.get("temperature", 1.0)
AG_GEN_TOP_K: int = _ag_gen.get("top_k", 250)
AG_GEN_CFG_COEF: float = _ag_gen.get("cfg_coef", 3.0)