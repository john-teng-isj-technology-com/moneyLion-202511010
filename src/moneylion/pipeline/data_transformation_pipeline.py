import os
from pathlib import Path
from src.moneylion import logger
from src.moneylion.components.data_transformation import DataTransformation
from src.moneylion.config.configuration import ConfigurationManager
from src.moneylion.utils.gcs_uploader import GSCUploader

class DataTransformationPipeline:

    def __init__(self) -> None:
        self.STAGE_NAME = "Data Transformation"
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_data_transformation_config()

    def upload_to_gcs(self):
        gcs_config = self.config_manager.get_gcs_artifact_config()
        obj = GSCUploader(gcs_config)
        obj.upload_directory_to_gcs(
            local_directory= Path(self.config.root_dir.name)
        )

    def initiate_data_transformation(self) -> bool:

        transformation_component = DataTransformation(config=self.config)
        transformation_component.transform_data()

        self.upload_to_gcs()

        return True

if __name__ == '__main__':
    try :
        obj = DataTransformationPipeline()
        logger.info(f">>>> Stage '{obj.STAGE_NAME}' started <<<<")
        obj.initiate_data_ingestion()
        logger.info(f">>>> Stage '{obj.STAGE_NAME}' completed successfully <<<<\n")
    except Exception as e:
        logger.exception(e)