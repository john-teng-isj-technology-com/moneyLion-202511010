import os
from pathlib import Path
from google.cloud import storage
from src.moneylion import logger
from src.moneylion.entity.config_entity import GCSArtifactConfig

class GSCUploader():
    def __init__(self, config: GCSArtifactConfig):
        self.config = config
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(config.gcp_credentials_path)
        self.storage_client = storage.Client()
            
    def upload_directory_to_gcs(self, local_directory: Path):
        try:
            bucket_name = self.config.bucket_name
            bucket = self.storage_client.bucket(bucket_name)
            gcs_prefix = f"{self.config.base_prefix}/{local_directory}"
            local_path = Path(gcs_prefix)
            logger.info(f"Starting upload from '{local_directory}' to 'gs://{bucket_name}/{gcs_prefix}'")

            for local_file in local_path.rglob("*"):
                if local_file.is_file():
                    relative_path = local_file.relative_to(local_path)
                    blob_name = os.path.join(gcs_prefix, str(relative_path))
                    
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(str(local_file))
            
            logger.info(f"Successfully uploaded artifacts to 'gs://{bucket_name}/{gcs_prefix}'")

        except Exception as e:
            logger.exception(f"Failed to upload directory '{local_directory}' to GCS. Error: {e}")
            raise
