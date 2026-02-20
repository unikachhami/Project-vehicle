import sys
from typing import  Tuple
from src.logger import logging
from src.exception import MyException
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score,accuracy_score,precision_score,f1_score
from src.utils.main_utils import load_object,save_numpy_array_data,save_object,load_numpy_array_data
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact,ClassificationMetricArtifact, ModelTrainerArtifact
from src.entity.estimator import MyModel



class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_config:ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
    def get_model_object_and_report(self,train:np.array,test:np.array)->Tuple[object,object]:
        try:
            logging.info("Model training with RandomForestClassifier")
            x_train,y_train,x_test,y_test = train[:,:-1],train[:,-1],test[:,:-1],test[:,-1]
            
            model = RandomForestClassifier(
                n_estimators=self.model_trainer_config._n_estimators,
                min_samples_split=self.model_trainer_config._min_samples_split,
                min_samples_leaf=self.model_trainer_config._min_samples_leaf,
                max_depth=self.model_trainer_config._max_depth,
                criterion=self.model_trainer_config._criterion,
                random_state = self.model_trainer_config._random_state



            )
            logging.info("Model Training Started:")
            model.fit(x_train,y_train)
            logging.info("Model Training Completed")

            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)

            metric_artifact= ClassificationMetricArtifact(f1_score=f1,precision_score=precision,recall_score=recall)
            return model,metric_artifact
        except Exception as e:
            raise MyException(e,sys) from e
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Train and Test array data loaded")
            logging.info("Entered initiate model artifact")
            trained_model,metric_artifact = self.get_model_object_and_report(train=train_arr,test=test_arr)
            logging.info("Model object loaded")
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            if accuracy_score(train_arr[:,-1],trained_model.predict(train_arr[:,:-1]))< self.model_trainer_config.expected_accuracy:
                logging.info("No model found with sscore above the base score")
                raise Exception("No model found with sscore above the base score")
            my_model = MyModel(preprocessing_object=preprocessing_obj,trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path,my_model)
            logging.info('Saved preprocessing object ad model')
            logging.info("Saving model:")

            model_trainer_artifact= ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )
            logging.info(f"Model trainer artifact {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise MyException(e,sys) from e









