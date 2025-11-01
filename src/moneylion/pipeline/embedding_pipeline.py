from src.moneylion import logger
from src.moneylion.components.embedding_export import EmbeddingExporter
from src.moneylion.config.configuration import ConfigurationManager

class EmbeddingPipeline:
    STAGE_NAME = "Data Embedding & Export"

    def __init__(self):
        self.cfg = ConfigurationManager().get_data_embedding_config()

    def initiate_embedding(self) -> bool:
        EmbeddingExporter(self.cfg).run()
        return True
