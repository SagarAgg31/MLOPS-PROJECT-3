from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
import os
import sys
from src.feature_store import RedisFeatureStore
import pandas as pd
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self,feature_store:RedisFeatureStore,model_save_path:str):
        self.feature_store = feature_store
        self.model_save_path = model_save_path
        self.model = None

        os.makedirs(self.model_save_path,exist_ok=True)
        logger.info("Model Training Intialized!!")

    def load_data_from_redis(self,entity_ids):
        try:
            logger.info("Extracting data from redis...")
            data = []
            for entity_id in entity_ids:
                features = self.feature_store.get_features(entity_id)
                if features:
                    data.append(features)
                else:
                    logger.warning("Features not found!")
            return data
        except Exception as e:
            logger.error(f"Erorr while loading the from redis {e}")
            raise CustomException("Error while loading the data from redis",sys)
        
    def prepare_data(self):
        try:
            entity_ids = self.feature_store.get_all_entity_ids()

            train_entity_ids,test_entity_ids = train_test_split(entity_ids,test_size=0.2,random_state=42)

            train_data = self.load_data_from_redis(train_entity_ids)
            test_data = self.load_data_from_redis(test_entity_ids)

            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)

            X_train = train_df.drop('Survived',axis=1)
            logger.info(X_train.columns)
            X_test = test_df.drop('Survived',axis=1)
            logger.info(X_test.columns)
            
            y_train = train_df['Survived']
            y_test = test_df['Survived']

            logger.info("Preparation for Model Training Completed!!")
            return X_train,X_test,y_train,y_test

        except Exception as e:
            logger.error(f"Error while preparing the data {e}")
            raise CustomException("Erorr while preparing the data",sys)
        
    def hyperparameter_tuning(self,X_train,y_train):
        try:
            param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
            }
            rf = RandomForestClassifier(random_state=42)
            random_search = RandomizedSearchCV(rf, param_distributions, n_iter=10, cv=3, scoring='accuracy', random_state=42)
            random_search.fit(X_train, y_train)

            logger.info(f"Best parameters : {random_search.best_estimator_}")
            return random_search.best_estimator_
        
        except Exception as e:
            logger.error("Error while HyperParameter Tuning")
            raise CustomException(str(e),sys)
        
    def train_evaluate(self,X_train,y_train,X_test,y_test):
        try:
            best_rf = self.hyperparameter_tuning(X_train, y_train)
            
            y_pred = best_rf.predict(X_test)

            accuracy = accuracy_score(y_test,y_pred)

            logger.info(f"Accuracy is: {accuracy}")

            self.save_model(best_rf)

            return accuracy

        except Exception as e:
            logger.error(f"Error while training and evaluation {e}")
            raise CustomException(str(e),sys)
        
    def save_model(self,model):
        try:
            model_file_name = f"{self.model_save_path}random_forest.pkl"

            with open(model_file_name,'wb') as model_file:
                pickle.dump(model,model_file)

            logger.info(f"Model Save at {model_file_name}")

        except Exception as e:
            logger.error(f"Error while saving the model {e}")
            raise CustomException("Error while saving the model",sys)
        
    def run(self):
        try:
            logger.info("Starting Model Training Pipeline....")
            X_train,X_test,y_train,y_test = self.prepare_data()
            accuracy = self.train_evaluate(X_train, y_train, X_test, y_test)

            logger.info("End of Model Training Pipeline...")

        except Exception as e:
            logger.error(f"Error while running the Model Training Pipeline {e}")
            raise CustomException("Error while running Model Training Pipeline",sys)

if __name__=='__main__':
    feature_store = RedisFeatureStore()
    model_trainer = ModelTraining(model_save_path=MODEL_PATH,feature_store=feature_store)
    model_trainer.run()


        
    
