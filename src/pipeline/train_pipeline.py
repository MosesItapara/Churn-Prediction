print("SCRIPT STARTED")
import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def run_pipeline(self):
        try:
            print("STEP 0: entering try block", flush=True)
            logging.info("Training pipeline started")

            print("STEP 1: starting data ingestion", flush=True)
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            print("STEP 1 DONE", flush=True)

            print("STEP 2: starting data transformation", flush=True)
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            print("STEP 2 DONE", flush=True)

            print("STEP 3: starting model training", flush=True)
            model_trainer = ModelTrainer()
            roc, f1 = model_trainer.initiate_model_trainer(train_arr, test_arr)
            print("STEP 3 DONE", flush=True)

            return roc, f1

        except Exception as e:
            print("EXCEPTION CAUGHT:", e, flush=True)
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    print("ABOUT TO CALL run_pipeline", flush=True)
    pipeline = TrainPipeline()
    roc, f1 = pipeline.run_pipeline()
    print(f"Training complete — ROC-AUC: {roc}, F1: {f1}", flush=True)