import yaml
from dataclasses import dataclass
from pathlib import Path

_CFG_PATH = Path(__file__).parent / "config.yaml"


def _load_yaml(path=_CFG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


_raw = _load_yaml()

DEVICE = _raw["device"]

SAMPLE_RATE = _raw["audio"]["sample_rate"]
CHUNK_LENGTH = _raw["audio"]["chunk_length"]
FADE_SEC = _raw["audio"]["fade_sec"]
MIN_CHUNK_SEC = _raw["audio"]["min_chunk_sec"]

_snac = _raw["snac"]
SNAC_MODEL = _snac["model"]
CODEBOOK_SIZE = _snac["codebook_size"]
SNAC_N_LEVELS = _snac["n_levels"]

_snac_gen = _snac.get("generation", {})
SNAC_GEN_TEMPERATURE = _snac_gen.get("temperature", 1.0)
SNAC_GEN_TOP_K = _snac_gen.get("top_k", 50)

_snac_inf = _snac.get("inference", {})
SNAC_INF_BATCH_SIZE = _snac_inf.get("batch_size", 32)
SNAC_INF_NUM_WORKERS = _snac_inf.get("num_workers", 8)

_data = _raw["data"]
TOKEN_DIR = Path(_data["token_dir"])
SEGMENT_DIR = Path(_data["segment_dir"])
VAL_RATIO = _data["val_ratio"]
TEST_RATIO = _data["test_ratio"]
SEED = _data["seed"]
MIN_SAMPLES_PER_CLASS = _data["min_samples_per_class"]

_pretrain = _raw["pretrain"]
PRETRAIN_EPOCHS = _pretrain["epochs"]
PRETRAIN_BATCH_SIZE = _pretrain["batch_size"]
PRETRAIN_LR = _pretrain["learning_rate"]
PRETRAIN_WARMUP = _pretrain["warmup_steps"]
MAX_SEQ_LEN = _pretrain["max_seq_len"]
PRETRAIN_NUM_WORKERS = _pretrain["num_workers"]
PRETRAIN_SAVE_DIR = Path(_pretrain["save_dir"])

_model = _raw["model"]
BACKBONE = _model.get("backbone", "gpt2")
N_EMBD = _model["n_embd"]
N_LAYER = _model["n_layer"]
N_HEAD = _model["n_head"]
N_POSITIONS = _model["n_positions"]
INTERMEDIATE_SIZE = _model.get("intermediate_size", None)

_wandb = _raw["wandb"]
WANDB_PROJECT = _wandb["project"]
WANDB_ENTITY = _wandb["entity"]

_ag = _raw.get("audiogen", {})
AG_PRETRAINED = _ag.get("pretrained", "facebook/audiogen-medium")
AG_SAMPLE_RATE = _ag.get("sample_rate", 16000)
AG_TARGET_LENGTH = _ag.get("target_length", 10)
AG_SAVE_DIR = Path(_ag.get("save_dir", "checkpoints/audiogen"))
AG_NUM_WORKERS = _ag.get("num_workers", 8)
AG_GRAD_ACCUM = _ag.get("grad_accum", 1)

_ag_tokens = _ag.get("encodec_tokens", {})
AG_TOKEN_DIR = _ag_tokens.get("dir", "data/encodec_tokens")

_ag_enc_inf = _ag.get("encodec_inference", {})
AG_ENCODEC_BATCH_SIZE = _ag_enc_inf.get("batch_size", 32)
AG_ENCODEC_NUM_WORKERS = _ag_enc_inf.get("num_workers", 8)


@dataclass
class StageConfig:
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: float = 32.0


_s1 = _ag.get("stage1", {})
AG_STAGE1 = StageConfig(
    epochs=_s1.get("epochs", 20),
    batch_size=_s1.get("batch_size", 8),
    learning_rate=_s1.get("learning_rate", 1e-3),
    warmup_steps=_s1.get("warmup_steps", 100),
)

_s2 = _ag.get("stage2", {})
_s2_lora = _s2.get("lora", {})
AG_STAGE2 = StageConfig(
    epochs=_s2.get("epochs", 50),
    batch_size=_s2.get("batch_size", 4),
    learning_rate=_s2.get("learning_rate", 1e-4),
    warmup_steps=_s2.get("warmup_steps", 1000),
    use_lora=_s2_lora.get("enabled", False),
    lora_rank=_s2_lora.get("rank", 16),
    lora_alpha=_s2_lora.get("alpha", 32.0),
)

_s3 = _ag.get("stage3", {})
_s3_lora = _s3.get("lora", {})
AG_STAGE3 = StageConfig(
    epochs=_s3.get("epochs", 50),
    batch_size=_s3.get("batch_size", 4),
    learning_rate=_s3.get("learning_rate", 5e-5),
    warmup_steps=_s3.get("warmup_steps", 500),
    use_lora=_s3_lora.get("enabled", False),
    lora_rank=_s3_lora.get("rank", 16),
    lora_alpha=_s3_lora.get("alpha", 32.0),
)

_ag_gen = _ag.get("generation", {})
AG_GEN_DURATION = _ag_gen.get("duration", 10.0)
AG_GEN_TEMPERATURE = _ag_gen.get("temperature", 1.0)
AG_GEN_TOP_K = _ag_gen.get("top_k", 250)
AG_GEN_CFG_COEF = _ag_gen.get("cfg_coef", 3.0)

_s3_cfg = _raw.get("s3", {})
S3_BUCKET = _s3_cfg.get("bucket", "ar-bird-vocalization")
S3_PREFIX = _s3_cfg.get("prefix", "tokens")

_bc = _raw.get("birdclef", {})
BC_DATA_DIR = Path(_bc.get("data_dir", "data/birdclef-2026"))
BC_SEGMENT_DIR = Path(_bc.get("segment_dir", "data/birdclef_segments"))
BC_EBIRD_TO_ID_PATH = Path(
    _bc.get("ebird_to_id_path", str(SEGMENT_DIR / "ebird_to_id.json"))
)
BC_MIN_SAMPLES_PER_CLASS = _bc.get("min_samples_per_class", 20)

_bc_gating = _bc.get("gating", {})
BC_GATING_MIN_TOP1_PROB = float(_bc_gating.get("min_top1_prob", 0.5))
BC_GATING_MAX_ENTROPY = float(_bc_gating.get("max_entropy", 2.0))
BC_GATING_BATCH_SIZE = int(_bc_gating.get("batch_size", 16))

_bc_xcm = _bc.get("xcm_enrich", {})
BC_XCM_ENRICH_ENABLED = bool(_bc_xcm.get("enabled", False))
BC_XCM_HF_PATH = str(_bc_xcm.get("hf_path", "DBD-research-group/BirdSet"))
BC_XCM_HF_NAME = str(_bc_xcm.get("hf_name", "XCM"))
_xcm_root = _bc_xcm.get("audio_root")
BC_XCM_AUDIO_ROOT = None if _xcm_root in (None, "", "null") else str(_xcm_root)
BC_XCM_METADATA_ONLY = bool(_bc_xcm.get("metadata_only", True))
BC_XCM_SHUFFLE_BUFFER_SIZE = int(_bc_xcm.get("shuffle_buffer_size", 50_000))
BC_XCM_MAX_STREAM_PASSES = int(_bc_xcm.get("max_stream_passes", 5))

_eval = _raw.get("evaluation", {})
EVAL_BATCH_SIZE = _eval.get("batch_size", 32)
EVAL_NUM_WORKERS = _eval.get("num_workers", 8)
EVAL_SAMPLE_RATE = _eval.get("sample_rate", 32000)
EVAL_MODEL_CHECKPOINT = _eval.get(
    "model_checkpoint", "DBD-research-group/ConvNeXT-Base-BirdSet-XCL"
)
EVAL_N_FFT = _eval.get("n_fft", 1024)
EVAL_HOP_LENGTH = _eval.get("hop_length", 320)
EVAL_N_MELS = _eval.get("n_mels", 128)
EVAL_SPEC_MEAN = _eval.get("spec_mean", -4.268)
EVAL_SPEC_STD = _eval.get("spec_std", 4.569)
EVAL_SPEC_SIZE = _eval.get("spec_size", (224, 224))