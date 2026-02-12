import os
from dataclasses import dataclass


DEFAULT_MODEL = os.getenv("P2D_DEFAULT_MODEL", "gpt-5-nano")


def _model(name: str) -> str:
    return os.getenv(name, DEFAULT_MODEL)


@dataclass(frozen=True)
class ModelConfig:
    anatomy: str = _model("P2D_MODEL_ANATOMY")
    metadata: str = _model("P2D_MODEL_METADATA")
    methods: str = _model("P2D_MODEL_METHODS")
    results: str = _model("P2D_MODEL_RESULTS")
    data_availability: str = _model("P2D_MODEL_DATA_AVAILABILITY")
    quality: str = _model("P2D_MODEL_QUALITY")
    metadata_enrichment: str = _model("P2D_MODEL_METADATA_ENRICHMENT")
    synthesis: str = _model("P2D_MODEL_SYNTHESIS")
    manager: str = _model("P2D_MODEL_MANAGER")
    harmonizer: str = _model("P2D_MODEL_HARMONIZER")


AGENT_VERSION = "0.1.0"
MODELS = ModelConfig()
