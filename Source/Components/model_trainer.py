import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV

from Source.exception import CustomException
from Source.logger import logging
from Source.utils import save_object,eval_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "LinearRegression" : LinearRegression(),
                "Lasso" : Lasso(),
                "Ridge" : Ridge(),
                "K-Neighbors Regressor" : KNeighborsRegressor(),
                "Decision Tree Regressor" : DecisionTreeRegressor(),
                "Random Forest Regressor" : RandomForestRegressor(),
                "Support Vector Regressor" : SVR(),
                "Ada Boost Regressor" : AdaBoostRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "CatBoost Regressor" : CatBoostRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor()
            }

            
            params= {

                "Random Forest Regressor" : {
                    "max_depth": [5, 8, 10, 15, None],
                    "max_features": [5, 7, "auto", "sqrt"],
                    "min_samples_split": [2, 5, 8, 15, 20],
                    "min_samples_leaf": [1, 2, 4],
                    "n_estimators": [100, 200, 500, 1000]
                },
                "XGBRegressor" : {
                    "learning_rate": [0.3, 0.1, 0.05, 0.01],
                    "max_depth": [3, 5, 8, 12],
                    "n_estimators": [100, 200, 500],
                    "colsample_bytree": [0.3, 0.5, 0.8, 1],
                    "subsample": [0.6, 0.8, 1]
                },
                "CatBoost Regressor" : {
                    "iterations": [200, 500, 1000],
                    "depth": [4, 6, 8, 10],
                    "learning_rate": [0.3, 0.1, 0.05, 0.01],
                    "l2_leaf_reg": [1, 3, 5, 7, 9]
                },
                "Support Vector Regressor" : {
                    "kernel": ["linear", "poly", "rbf", "sigmoid"],
                    "C": [0.1, 1, 10, 100],
                    "epsilon": [0.01, 0.1, 0.2, 0.5],
                    "gamma": ["scale", "auto"]
                },
                "K-Neighbors Regressor" : {
                    "n_neighbors": [3, 5, 7, 9, 15],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan", "minkowski"]
                },
                "Decision Tree Regressor" : {
                    "max_depth": [3, 5, 8, 12, None],
                    "min_samples_split": [2, 5, 10, 15],
                    "min_samples_leaf": [1, 2, 4, 6],
                    "max_features": [None, "sqrt", "log2"]
                },
                "Lasso" : {
                    "alpha": [0.001, 0.01, 0.1, 1, 5, 10],
                    "max_iter": [500, 1000, 5000]
                },
                "Ridge" : {
                    "alpha": [0.001, 0.01, 0.1, 1, 5, 10, 50],
                    "max_iter": [500, 1000, 5000]
                },
                "LinearRegression" : {
                    "fit_intercept": [True, False],
                    "positive": [True, False]
                },
                "Ada Boost Regressor" : {
                    "n_estimators": [50, 100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1],
                    "loss": ["linear", "square", "exponential"]
                },
                "Gradient Boosting" : {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.3, 0.1, 0.05, 0.01],
                    "max_depth": [3, 5, 8],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "subsample": [0.6, 0.8, 1.0],
                    "max_features": [None, "sqrt", "log2"],
                    "loss": ["squared_error", "absolute_error", "huber"]
                }
            }


            model_report:dict=eval_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,models=models,param=params)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best Model Found.")
            
            logging.info(f"Best Found Model on Both Training & Testing Dataset.")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
