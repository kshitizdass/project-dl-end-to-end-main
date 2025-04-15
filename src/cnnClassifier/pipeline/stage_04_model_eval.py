import sys
import os
sys.path.append(os.getcwd())

from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.model_eval import Evaluation
from src.cnnClassifier import logger
import mlflow
import os



STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/kshitiz.dass/project-dl-end-to-end-main.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"]="kshitiz.dass"
        os.environ["MLFLOW_TRACKING_PASSWORD"]="0f6ed27c58b2835ecaea6db1197d2f59acc980f6"

        mlflow.set_tracking_uri("https://dagshub.com/kshitiz.dass/project-dl-end-to-end-main.mlflow")
        evaluation.log_into_mlflow()




if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e