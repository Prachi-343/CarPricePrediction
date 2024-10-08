from src.CarPricePrediction.components.data_ingestion import DataIngestion
from src.CarPricePrediction.components.data_transformation import DataTransformation
from src.CarPricePrediction.components.model_trainer import ModelTrainer

import os
import sys
import pandas as pd
import numpy as np
from src.CarPricePrediction.logger import logging
from src.CarPricePrediction.exception import customexception

class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            data_ingest=DataIngestion()
            train_data_path,test_data_path=data_ingest.initiate_data_ingestion()
            return train_data_path,test_data_path
        
        except Exception as e:
            raise customexception(e,sys)
    
    def start_data_transformation(self,train_data_path,test_data_path):
        try:
            data_tansformation=DataTransformation()
            train_arr,test_arr=data_tansformation.initiate_data_transformation(train_data_path,test_data_path)
            return train_arr,test_arr
        
        except Exception as e:
            raise customexception(e,sys)
        
    def start_model_training(self,train_arr,test_arr):
        try:
            model_trainer=ModelTrainer()
            model_trainer.initiate_model_training(train_arr,test_arr) # Assuming the method name is correct
            
        except Exception as e:
            raise customexception(e,sys)
        
    def start_traning(self):
        try:
            train_data_path,test_data_path=self.start_data_ingestion()
            train_arr,test_arr=self.start_data_transformation(train_data_path,test_data_path)
            self.start_model_training(train_arr,test_arr)
        except Exception as e:
            raise customexception(e,sys)

if __name__ == "__main__":
    training_obj=TrainingPipeline() # Corrected the object instantiation
    training_obj.start_traning()    
        
        