import numpy as np
import pandas as pd 
import os
import sys

from src.CarPricePrediction.utils.utils import save_object
from src.CarPricePrediction.logger import logging
from src.CarPricePrediction.exception import customexception
from dataclasses import dataclass
from pathlib import Path

from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler  # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class DataTransformationconfig:
    preprocessor_object_file_path=os.path.join("artifacts","preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()
        
    def get_data_transformation(self):
        logging.info("getting data transformation")
        
        try:
            numerical_cols=['annual Salary', 'credit card debt', 'net worth']
            
            num_pipelines= Pipeline(
                steps=[
                    ("impute",SimpleImputer()),
                    ("scaler",StandardScaler())
                ]
            )
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipelines",num_pipelines,numerical_cols)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            logging.info("fail to get data transformation")
            raise(e,sys)

    def initate_data_transformations(self,train_data_path,test_data_path):
        try:
            train_data=pd.read_csv(train_data_path)
            test_data=pd.read_csv(test_data_path)
            
            logging.info("completed train and test reading")
            logging.info(f"train_data \n {train_data.head().to_string()}")
            logging.info(f"test_data \n {test_data.head().to_string()}")
            
            preprocessor_obj=self.get_data_transformation()
            
            target_column_name="car purchase amount"
            drop_columns=[target_column_name,"customer name","customer e-mail","country","gender","age"]

            input_features_train_data=train_data.drop(columns=drop_columns,axis=1)
            target_features_train_data=train_data[target_column_name]
            
            input_featurs_test_data=test_data.drop(columns=drop_columns,axis=1)
            target_features_test_data=test_data[target_column_name]
            
            input_features_train_arr=preprocessor_obj.fit_transform(input_features_train_data)
            input_features_test_arr=preprocessor_obj.transform(input_featurs_test_data)
            
            logging.info("perprocessing on training and testing data")
            
            train_arr=np.c_[input_features_test_arr,np.array(target_features_train_data)]
            test_arr=np.c_[input_features_test_arr,np.array(target_features_test_data)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            return(
                train_arr,test_arr
            )
                    
        except Exception as e:
            logging.info("fail to get initiate data transformations")
            raise(e,sys)
        
        

