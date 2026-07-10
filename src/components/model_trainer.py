import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import src

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "XGBClassifier": XGBClassifier()
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)

            # Find the NAME of the best model, not just its score
            best_model_name = max(model_report, key=lambda name: model_report[name]["test_model_score"])
            best_model_score_value = model_report[best_model_name]["test_model_score"]

            # Now index `models` by the actual model name — this is the fix
            best_model = models[best_model_name]

            logging.info(f"Best model found on both training and testing dataset: {best_model_name} with score: {best_model_score_value}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            roc = roc_auc_score(y_test, predicted)
            f1 = f1_score(y_test, predicted)
            return roc, f1
        
        except Exception as e:
            raise CustomException(e, sys)