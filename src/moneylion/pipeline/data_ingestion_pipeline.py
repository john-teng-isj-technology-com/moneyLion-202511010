# This library loads environment variables from a .env file
from dotenv import load_dotenv

# Import project-specific modules
from src.moneylion import logger
from src.moneylion.components.data_ingestion import DataIngestion
from src.moneylion.config.configuration import ConfigurationManager


class DataIngestionPipeline:

    def __init__(self) -> None:
        self.config_manager = ConfigurationManager()
        self.STAGE_NAME = "Data Ingestion"

    def initiate_data_ingestion(self) -> bool:
        load_dotenv()
        logger.info("Loaded environment variables from .env file.")

        # Get the configuration for the data ingestion stage
        ingestion_config = self.config_manager.get_data_ingestion_config()

        # Instantiate and run the component
        ingestion_component = DataIngestion(config=ingestion_config)
        ingestion_component.download_files()

        return True

if __name__ == '__main__':
    try :
        obj = DataIngestionPipeline()
        logger.info(f">>>> Stage '{obj.STAGE_NAME}' started <<<<")
        obj.initiate_data_ingestion()
        logger.info(f">>>> Stage '{obj.STAGE_NAME}' completed successfully <<<<\n")
    except Exception as e:
        logger.exception(e)