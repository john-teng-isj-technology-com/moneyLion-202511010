from dataclasses import dataclass
from typing import Callable, Type, Any, Optional
from src.moneylion import logger

# Import your existing pipeline classes
from src.moneylion.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.moneylion.pipeline.data_transformation_pipeline import DataTransformationPipeline


@dataclass(frozen=True)
class StageSpec:
    pipeline_cls: Type[Any]
    method_name: str
    require_true: bool = False  # set True for validation-like stages


class MainSequence:
    def __init__(self):
        # Define the sequence once; easy to reorder or extend
        self.stages: list[StageSpec] = [
            # StageSpec(DataIngestionPipeline, 'initiate_data_ingestion'),
            StageSpec(DataTransformationPipeline, 'initiate_data_transformation'),
        ]

    def _run_stage(self, spec: StageSpec) -> Optional[Any]:
        obj = spec.pipeline_cls()
        stage_name = getattr(obj, 'STAGE_NAME', obj.__class__.__name__)

        logger.info(f'>>>> stage {stage_name} started <<<<')
        result = getattr(obj, spec.method_name)()
        if spec.require_true and not result:
            # keep your original behavior for validation failure
            raise Exception('VALIDATION FAILED')
        logger.info(f'>>>> stage {stage_name} completed <<<<')
        return result

    def run(self) -> None:
        try:
            for spec in self.stages:
                self._run_stage(spec)
        except Exception as e:
            logger.exception(e)
            raise


if __name__ == '__main__':
    main_sequence = MainSequence()
    main_sequence.run()
