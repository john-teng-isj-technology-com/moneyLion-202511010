from src.moneylion import logger
from src.moneylion.components.model_trainer import XGBTrainer
from src.moneylion.config.configuration import ConfigurationManager
from src.moneylion.utils.gcs_uploader import GSCUploader
from pathlib import Path


class ModelTrainingPipeline:
    STAGE_NAME = "Model Training"

    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_model_training_config()

    def upload_to_gcs(self):
        gcs_config = self.config_manager.get_gcs_artifact_config()
        obj = GSCUploader(gcs_config)
        obj.upload_directory_to_gcs(
            local_directory= Path(self.config.root_dir.name)
        )

    def initiate_model_training(self) -> bool:
        XGBTrainer(self.config).run()
        self.upload_to_gcs()
        return True
