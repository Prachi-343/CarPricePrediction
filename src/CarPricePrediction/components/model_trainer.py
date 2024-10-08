import os
import sys
import numpy as np
import pandas as pd

from src.CarPricePrediction.logger  import logging
from src.CarPricePrediction.exception import customexception
from src.CarPricePrediction.utils.utils import save_object
from src.CarPricePrediction.utils.utils import evaluate_model
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet

@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join("artifacts","model.pkl")
class   ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer_config(self,train_array,test_array):
        logging.info("model traning started")
        try:
            logging.info("splitting dependent and independent variable in train and test data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                
            )
            
            models={
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet()
            }
            
            model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,models)
            print(model_report)
            print("\n================================\n")
            logging.info(f"model_report:\n{model_report}")
            
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            print(f"Best Model Found,Model Name:{best_model_name},R2 Score:{best_model_score}")
            print("\n================================\n")
            logging.info(f"Best Model Found,Model Name:{best_model_name},R2 Score:{best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )
            
        except Exception as e:
            logging.info("error in model traning")
            raise(e,sys)