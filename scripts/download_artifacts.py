from pathlib import Path

import boto3
import typer
from settings import Settings


def download_models(settings: Settings) -> None:
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(settings.artifact_bucket_name)

    if not bucket.creation_date:
        print(f"Bucket '{settings.artifact_bucket_name}' does not exist.")
        return

    print(f"Starting download from bucket: {settings.artifact_bucket_name}")

    directory = settings.onnx_classifier_path.parent
    directory.mkdir(parents=True, exist_ok=True)

    for obj in bucket.objects.all():
        local_file_path = directory / str(obj.key)
        local_file_dir = local_file_path.parent
        local_file_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {obj.key} to {str(local_file_path)}")
        bucket.download_file(obj.key, local_file_path)

    print("Download complete.")


def main(settings_path: Path) -> None:
    settings = Settings.from_yaml(settings_path)
    download_models(settings)


if __name__ == "__main__":
    typer.run(main)
