# pipeline/data_preprocessing_pipeline.py
from src.moneylion import logger
from src.moneylion.components.data_preprocessing import DataPreprocessor
from src.moneylion.config.configuration import ConfigurationManager

class DataPreprocessingPipeline:
    STAGE_NAME = "Data Preprocessing"

    def __init__(self) -> None:
        cfg_mgr = ConfigurationManager()
        self.cfg  = cfg_mgr.get_data_preprocessing_config()

    def initiate_data_preprocessing(self) -> bool:
        DataPreprocessor(self.cfg).run()
        return True
