import numpy as np
import pandas as pd      
import os
import sys 

from sklearn.model_selection import train_test_split
from src.CarPricePrediction.logger import logging
from src.CarPricePrediction.exception import customexception
from dataclasses import dataclass
from pathlib import Path


class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    tesrt_data_path:str=os.path.join("artifacts","test.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        
        logging.info("DataIngestion started")
        try:
            data=pd.read_csv(Path(os.join("notebooks\data","carprice.csv")))
            logging.info("data reading completed")
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("save raw data in artifacts folder")
            
            train_data,test_data=train_test_split(data,test_size=0.25)
            logging.info("train test split completed")
            
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info("data ingestion completed")
           
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
           
            
        except Exception as e:
            logging.info("error occur in data ingestion")
            raise customexception (e,sys)    
obj=DataIngestion()
obj.initiate_data_ingestion()
