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

            model_report:dict=eval_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,models=models)
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
