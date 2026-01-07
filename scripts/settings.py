from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel


class Settings(BaseModel):
    sentence_transformer_dir: Path
    classifier_joblib_path: Path
    onnx_classifier_path: Path
    onnx_embedding_model_path: Path
    tokenizer_json_path: Path

    embedding_dim: int

    artifact_bucket_name: str
    image_repository_name: str

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        content = yaml.safe_load(path.open())
        return cls(**content)
