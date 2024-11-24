import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath = os.path.join('artifacts' , 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()  # we will get preprocessor_obj_filepath in this variable

    def get_transformer_obj(self):      # For creating pickle files which will transform our data
        try:
            num_cols = ['writing_score' , 'reading_score']
            cat_cols = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps = [
                    ('Imputer' , SimpleImputer(strategy='median')),
                    ('scaler' , StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('Imputer' , SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoding' , OneHotEncoder()),
                    ('Scaler' , StandardScaler(with_mean=False))
                ]
            )

            preprocessor =ColumnTransformer(
                [
                    ('num_pipeline' , num_pipeline , num_cols),
                    ('cat_pipeline' , cat_pipeline , cat_cols)
                ]
            )
            logging.info('Pipeline for preprocssing build')

            return preprocessor
        
        except Exception as e:
            raise CustomException(e , sys)
        
    def initiate_data_transformation(self , train_path , test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data')

            logging.info('Obtaining preprocessing object')
            preprocessor_obj = self.get_transformer_obj()

            target_col_name = 'math_score'
            num_cols = ['writing_score' , 'reading_score']
            
            input_train_df = train_df.drop(target_col_name , axis=1)
            target_train_df = train_df[target_col_name]

            input_test_df = test_df.drop(target_col_name , axis=1)
            target_test_df = test_df[target_col_name]

            logging.info('Appplying preprocessing on training and testing data')

            input_train_trans = preprocessor_obj.fit_transform(input_train_df)
            input_test_trans = preprocessor_obj.transform(input_test_df)

            train_trans = np.c_[input_train_trans , np.array(target_train_df)]
            test_trans = np.c_[input_test_trans] , np.array(target_test_df)

            logging.info('Completed Preprocssing')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_filepath,
                obj = preprocessor_obj
            )
            logging.info('Saved preprocessing object')

            return (
                train_trans,
                test_trans,
                self.data_transformation_config.preprocessor_obj_filepath
            )
        except Exception as e:
            raise CustomException(e , sys)