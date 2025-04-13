import os

## DATA INGESTION
RAW_DIR = 'artifacts/raw'
TRAIN_PATH = os.path.join(RAW_DIR,'titanic_train.csv')
TEST_PATH = os.path.join(RAW_DIR,'titanic_test.csv')


# DATA PROCESSING
PROCESSED_DIR = 'artifacts/processed'

# MODEL TRAINING
MODEL_PATH = 'artifacts/models/'