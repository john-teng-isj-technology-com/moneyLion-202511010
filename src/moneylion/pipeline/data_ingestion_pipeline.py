import os
from pathlib import Path
from dotenv import load_dotenv
from src.moneylion import logger
from src.moneylion.components.data_ingestion import DataIngestion
from src.moneylion.config.configuration import ConfigurationManager


class DataIngestionPipeline:

    def __init__(self) -> None:
        self.config_manager = ConfigurationManager()
        self.STAGE_NAME = "Data Ingestion"

    def initiate_data_ingestion(self) -> bool:

        # Get the configuration for the data ingestion stage
        ingestion_config = self.config_manager.get_data_ingestion_config()
        cred_path = ingestion_config.gcp_credentials_path
        if cred_path and Path(cred_path).exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)
            logger.info(f"Using credentials from file: {cred_path}")
        else:
            logger.info("Credential file not found. Using Application Default Credentials (ADC).")

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