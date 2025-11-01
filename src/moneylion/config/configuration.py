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
        params = self.params.data_embedding
        create_directories([cfg.root_dir])
        return DataEmbeddingConfig(
            root_dir            = Path(cfg.root_dir),
            preproc_dir         = Path(cfg.preproc_dir),
            train_cat           = str(cfg.train_cat),
            train_num           = str(cfg.train_num),
            train_y             = str(cfg.train_y),
            val_cat             = str(cfg.val_cat),
            val_num             = str(cfg.val_num),
            val_y               = str(cfg.val_y),
            vocabs              = str(cfg.vocabs),
            embed_matrices      = str(cfg.embed_matrices),
            embed_schema        = str(cfg.embed_schema),
            epochs              = int(params.epochs),
            batch_size_train    = int(params.batch_size_train),
            batch_size_infer    = int(params.batch_size_infer),
            lr                  = float(params.lr),
            random_state        = int(params.random_state),
            embedding_dim_rule  = str(params.embedding_dim_rule),
        )
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        cfg = self.config.model_training
        params = self.params.model_training
        create_directories([cfg.root_dir])
        return ModelTrainingConfig(
            root_dir               = Path(cfg.root_dir),
            embed_dir              = Path(cfg.embed_dir),
            model_type             = params.model_type,
            param_grid             = params.param_grid,
            early_stopping_rounds  = int(params.early_stopping_rounds),
            metric                 = params.metric,
            decision_threshold     = str(params.decision_threshold),
            mlflow_experiment      = str(cfg.mlflow_experiment),
        )

