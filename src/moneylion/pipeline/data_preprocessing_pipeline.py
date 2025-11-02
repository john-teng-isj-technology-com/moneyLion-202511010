# pipeline/data_preprocessing_pipeline.py
from pathlib import Path
from src.moneylion import logger
from src.moneylion.components.data_preprocessing import DataPreprocessor
from src.moneylion.config.configuration import ConfigurationManager
from src.moneylion.utils.gcs_uploader import GSCUploader

class DataPreprocessingPipeline:
    STAGE_NAME = "Data Preprocessing"

    def __init__(self) -> None:
        self.config_manager = ConfigurationManager()
        self.config  = self.config_manager.get_data_preprocessing_config()

    def upload_to_gcs(self):
        gcs_config = self.config_manager.get_gcs_artifact_config()
        obj = GSCUploader(gcs_config)
        obj.upload_directory_to_gcs(
            local_directory= Path(self.config.root_dir.name)
        )

    def initiate_data_preprocessing(self) -> bool:
        DataPreprocessor(self.config).run()
        self.upload_to_gcs()
        return True
