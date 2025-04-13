from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.feature_store import RedisFeatureStore
from src.model_training import ModelTraining
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from config.database_config import *

if __name__=="__main__":

    data_ingestion = DataIngestion(DB_CONFIG,RAW_DIR)
    data_ingestion.run()

    feature_store = RedisFeatureStore()

    data_processor = DataProcessing(TRAIN_PATH,TEST_PATH,feature_store)
    data_processor.run()

    model_trainer = ModelTraining(model_save_path=MODEL_PATH,feature_store=feature_store)
    model_trainer.run()