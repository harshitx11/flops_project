"""Centralized, environment-driven project settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"


class Settings(BaseSettings):
    """Runtime configuration shared by training and serving."""

    data_dir: Path = PROJECT_ROOT / "data"
    artifact_dir: Path = PROJECT_ROOT / "artifacts"
    report_dir: Path = PROJECT_ROOT / "reports"
    dataset_url: str = (
        "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
    )
    dataset_filename: str = "sentiment_sentences.csv"
    model_filename: str = "sentiment_pipeline.joblib"
    model_version: str = "1.0.0"
    random_seed: int = 42
    min_accepted_f1: float = 0.75
    max_text_length: int = 5000
    cors_origins: str = "http://localhost:8080,http://127.0.0.1:8080"
    model_path: Path | None = None

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SENTIMENT_", extra="ignore")

    def resolved_model_path(self) -> Path:
        return self.model_path or self.artifact_dir / self.model_filename

    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
