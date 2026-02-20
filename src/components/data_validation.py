import json 
import sys
import os
import pandas as pd
import numpy as np
from pandas import DataFrame
from src.logger import logging
from src.exception import MyException
from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from src.utils.main_utils import read_yaml_file,write_yaml_file
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e,sys) 
    
    def validate_number_of_columns(self,dataframe:DataFrame)->bool:
        try:
            status = len(dataframe.columns)== len(self._schema_config["columns"])
            logging.info(f"Is Required column present [{status}]")
            return status
        except Exception as e:
            raise MyException(e,sys)
    def is_column_exist(self,df:DataFrame) ->bool:
        try:
            dataframe_columns= df.columns
            missing_numerical_columns = []
            missing_categorical_column = []
            for column in self._schema_config['numerical_columns']:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)
            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")

            for column in self._schema_config['categorical_column']:
                if column not in dataframe_columns:
                    missing_categorical_column.append(column)
            
            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column: {missing_categorical_column}")
            
            return False if len(missing_numerical_columns)>0 or len(missing_categorical_column)>0 else True

        except Exception as e:
            raise MyException(e,sys) from e
        
    @staticmethod
    def read_data(file_path: str)->DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            validation_error_message = ""
            logging.info("Starting data validation")
            train_df,test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.train_file_path),
                                DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))
            
            status = self.validate_number_of_columns(dataframe=train_df)
            if not status:
                validation_error_message+= f"Columns are missing in train dataframe:"
            else:
                logging.info(f"All columns are present: {status}")


            status = self.validate_number_of_columns(dataframe=test_df)
            if not status:
                validation_error_message+= f"Columns are missing in test dataframe:"
            else:
                logging.info(f"All columns are present: {status}")

            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_message+= f"Columns are missing in train dataframe:"
            else:
                logging.info(f"All columns are present: {status}")


            
            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_message+= f"Columns are missing in test dataframe:"
            else:
                logging.info(f"All columns are present: {status}")
            
            validation_status= len(validation_error_message)==0

            data_validation_artifact = DataValidationArtifact(validation_status=validation_status,
                                                              message=validation_error_message,validation_report_file_path = self.data_validation_config.validation_report_file_path)

                
                
            report_path = self.data_validation_config.validation_report_file_path
            report_dir = os.path.dirname(report_path)

            os.makedirs(report_dir, exist_ok=True)

            validation_report = {
                'validation_status': validation_status,
                'message': validation_error_message.strip()
            }

            with open(report_path, "w") as report_file:
             json.dump(validation_report, report_file, indent=4)

            logging.info(f"Data validation file created at: {report_path}")
            logging.info(f"Report exists: {os.path.exists(report_path)}")
            logging.info(f"Data validation artifact: {data_validation_artifact}")


            return data_validation_artifact
        except Exception as e:
            raise MyException(e,sys)
    
    


    




            


        
    



