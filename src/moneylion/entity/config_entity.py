@dataclass
class DataIngestionConfig:
    root_dir: Path
    gcs_bucket: str
    raw_files: list[str]
    local_download_dir: Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    loan_raw: Path
    clarity_raw: Path
    joined_local: Path

@dataclass
class DataLoadingConfig:
    root_dir: Path
    local_file: Path
    gcs_target: str
