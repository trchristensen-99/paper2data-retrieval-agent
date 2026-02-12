from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    metadata: str = "gpt-4.1-mini"
    methods: str = "gpt-4.1"
    results: str = "gpt-4.1"
    data_availability: str = "gpt-4.1"
    quality: str = "gpt-4.1-mini"
    synthesis: str = "gpt-4.1"
    manager: str = "gpt-4.1-mini"


AGENT_VERSION = "0.1.0"
MODELS = ModelConfig()
