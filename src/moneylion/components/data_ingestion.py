from __future__ import annotations
import os
from pathlib import Path
from google.cloud import storage
from src.moneylion import logger
from src.moneylion.entity.config_entity import DataIngestionConfig
from src.moneylion.utils.common import create_directories


class DataIngestion:

    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config
        # Ensure the destination directory exists before any downloads
        create_directories([self.config.local_download_dir])
        self.storage_client = storage.Client()

    def download_files(self) -> None:
        # Get a client handle to the GCS bucket
        bucket = self.storage_client.bucket(self.config.gcs_bucket_name)
        logger.info(f"Successfully connected to GCS bucket: '{self.config.gcs_bucket_name}'")

        for filename in self.config.raw_files:
            local_path = os.path.join(self.config.local_download_dir, filename)
            blob_path = os.path.join(self.config.gcs_source_folder, filename)
            blob = bucket.blob(blob_path)
            blob.download_to_filename(local_path)
            logger.info(f"Successfully downloaded: '{filename}' from GCS to {local_path}")
