from src.moneylion.constants import *
from src.moneylion.utils.common import read_yaml, create_directories
from src.moneylion.entity.config_entity import (
    DataIngestionConfig, DataTransformationConfig, DataPreprocessingConfig, DataEmbeddingConfig, ModelTrainingConfig
)
from pathlib import Path

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
            gcs_bucket_name=config.gcs_bucket_name,
            gcs_source_folder = config.gcs_source_folder,
            raw_files=config.raw_files,
            local_download_dir=config.local_download_dir,
            # absolute path
            gcp_credentials_path = Path(config.gcp_credentials_path).resolve(), 
        )
        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        return DataTransformationConfig(
            root_dir        = config.root_dir,
            loan_raw        = config.loan_raw,
            clarity_raw     = config.clarity_raw,
            joined_local    = config.joined_local
        )

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        cfg = self.config.data_preprocessing
        create_directories([cfg.root_dir])
        return DataPreprocessingConfig(
            root_dir=Path(cfg.root_dir),
            joined_csv=Path(cfg.joined_csv),
            test_size=float(cfg.test_size),
            val_size=float(cfg.val_size),
            random_state=int(cfg.random_state),
        )

    def get_data_embedding_config(self) -> DataEmbeddingConfig:
        cfg = self.config.data_embedding
        create_directories([cfg.root_dir])
        return DataEmbeddingConfig(
            root_dir          = Path(cfg.root_dir),
            preproc_dir       = Path(cfg.preproc_dir),
            epochs            = int(cfg.epochs),
            batch_size        = int(cfg.batch_size),
            lr                = float(cfg.lr),
            random_state      = int(cfg.random_state),
            embedding_dim_rule= str(cfg.embedding_dim_rule),
        )
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        cfg = self.config.model_training
        create_directories([cfg.root_dir])
        return ModelTrainingConfig(
            root_dir               = Path(cfg.root_dir),
            embed_dir              = Path(cfg.embed_dir),
            model_type             = cfg.model_type,
            param_grid             = cfg.param_grid,
            early_stopping_rounds  = int(cfg.early_stopping_rounds),
            metric                 = cfg.metric,
        )

