from src.moneylion import logger
from src.moneylion.components.model_trainer import XGBTrainer
from src.moneylion.config.configuration import ConfigurationManager

class ModelTrainingPipeline:
    STAGE_NAME = "Model Training"

    def __init__(self):
        cfg_mgr = ConfigurationManager()
        self.cfg = cfg_mgr.get_model_training_config()

    def initiate_model_training(self) -> bool:
        XGBTrainer(self.cfg).run()
        return True
