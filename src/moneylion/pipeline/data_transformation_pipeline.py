import os
from pathlib import Path
from src.moneylion import logger
from src.moneylion.components.data_transformation import DataTransformation
from src.moneylion.config.configuration import ConfigurationManager


class DataTransformationPipeline:

    def __init__(self) -> None:
        self.config_manager = ConfigurationManager()
        self.STAGE_NAME = "Data Transformation"

    def initiate_data_transformation(self) -> bool:

        # Get the configuration for the data ingestion stage
        config = self.config_manager.get_data_transformation_config()

        # Instantiate and run the component
        transformation_component = DataTransformation(config=config)
        transformation_component.transform_data()

        return True

if __name__ == '__main__':
    try :
        obj = DataTransformationPipeline()
        logger.info(f">>>> Stage '{obj.STAGE_NAME}' started <<<<")
        obj.initiate_data_ingestion()
        logger.info(f">>>> Stage '{obj.STAGE_NAME}' completed successfully <<<<\n")
    except Exception as e:
        logger.exception(e)