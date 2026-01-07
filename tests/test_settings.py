from pathlib import Path

from scripts.settings import Settings


def test_settings() -> None:
    mock_config_path = Path(__file__).parent / "mock_config.yml"
    _ = Settings.from_yaml(mock_config_path)
