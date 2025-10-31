from src.moneylion.constants import *
from src.moneylion.utils.common import read_yaml, create_directories
from src.moneylion.entity.config_entity import (DataIngestionConfig, DataTransformationConfig, DataLoadingConfig)

class ConfigurationManager:
    def __init__(self, 
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH
                 ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            gcs_bucket=config.gcs_bucket,
            raw_files=config.raw_files,
            local_download_dir=config.local_download_dir
        )
        return data_ingestion_config