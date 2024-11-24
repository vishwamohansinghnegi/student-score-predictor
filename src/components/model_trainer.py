import os
import sys

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor , AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from dataclasses import dataclass
from src.logger import logging
from src.utils import evaluate_model
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_filepath = os.path.join('artifacts' , 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self , train_trans , test_trans):
        try:
            logging.info('Splitting training and test input data')
            X_train , y_train , X_test , y_test = (train_trans[:,:-1] ,
                                                   train_trans[:,-1],
                                                   test_trans[:,:-1] , 
                                                   test_trans[:,-1])
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                    "Random Forest": {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                    },
                    "Decision Tree": {
                        "criterion": ["squared_error", "friedman_mse"],
                        "max_depth": [10, 20, 30],
                        "min_samples_split": [2, 5],
                    },
                    "Gradient Boosting": {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.01, 0.1],
                        "max_depth": [3, 5],
                    },
                    "Linear Regression": {
                        "fit_intercept": [True, False],
                    },
                    "XGBRegressor": {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.01, 0.1],
                        "max_depth": [3, 5],
                    },
                    "CatBoosting Regressor": {
                        "iterations": [100, 200],
                        "learning_rate": [0.01, 0.1],
                        "depth": [6, 10],
                    },
                    "AdaBoost Regressor": {
                        "n_estimators": [50, 100],
                        "learning_rate": [0.01, 0.1],
                    },
            }

            model_report : dict=evaluate_model(X_train=X_train , y_train=y_train , X_test=X_test , y_test=y_test , models=models , params=params)

            # For best model score from model_report dict
            best_model_score = max(model_report.values())

            # For best model name from model_report dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No best model')
            
            logging.info('Best model found')

            predicted = best_model.predict(X_test)

            r2 = r2_score(y_test , predicted)

            return r2
        
        except Exception as e:
            raise CustomException(e , sys)