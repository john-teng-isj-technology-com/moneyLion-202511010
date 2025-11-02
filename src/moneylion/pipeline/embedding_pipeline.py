from src.moneylion import logger
from pathlib import Path
from src.moneylion.components.embedding_export import EmbeddingExporter
from src.moneylion.config.configuration import ConfigurationManager
from src.moneylion.utils.gcs_uploader import GSCUploader

class EmbeddingPipeline:
    STAGE_NAME = "Data Embedding & Export"

    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_data_embedding_config()

    def upload_to_gcs(self):
        gcs_config = self.config_manager.get_gcs_artifact_config()
        obj = GSCUploader(gcs_config)
        obj.upload_directory_to_gcs(
            local_directory= Path(self.config.root_dir.name)
        )

    def initiate_embedding(self) -> bool:
        EmbeddingExporter(self.config).run()
        self.upload_to_gcs()
        return True
