from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    gcs_bucket_name: str
    gcs_source_folder: str
    raw_files: list[str]
    local_download_dir: Path
    gcp_credentials_path: Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    loan_raw: Path
    clarity_raw: Path
    joined_local: Path
    dummy_cols_path: Path
    num_stats_path: Path

@dataclass
class DataLoadingConfig:
    root_dir: Path
    local_file: Path
    gcs_target: str

@dataclass
class DataPreprocessingConfig:
    root_dir: Path
    joined_csv: Path
    test_size: float
    val_size: float
    random_state: int 

@dataclass
class DataEmbeddingConfig:
    root_dir: Path
    preproc_dir: Path
    train_cat: str
    train_num: str
    train_y: str
    val_cat: str
    val_num: str
    val_y: str
    vocabs: str
    embed_matrices: str
    embed_schema: str
    epochs: int
    batch_size_train: int
    batch_size_infer: int
    lr: float
    random_state: int
    embedding_dim_rule: str  

@dataclass
class ModelTrainingConfig:
    root_dir: Path
    embed_dir: Path
    model_type: str
    param_grid: dict
    early_stopping_rounds: int
    metric: str
    decision_threshold: str         
    mlflow_experiment: str | None = None
