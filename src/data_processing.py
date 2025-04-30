import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection  import train_test_split
import os

from src.logger import get_logger
from src.exceptions import CustomException

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self,file_path):
        self.file_path= file_path
        self.df = None
        self.processed_data_path= "artifacts/processed"
        os.makedirs(self.processed_data_path, exist_ok=True)

    def load_data(self):
        try:
            self.df =pd.read_csv(self.file_path)
            logger.info("Data Loaded SUcessfully..")

        except Exception as e:
            logger.error(f"Error while loading the data: {e}")
            raise CustomException("Failed to load data", e)


    def handle_outliers(self,column):
        try:
            logger.info("starting Handle ouliers operation")
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3-Q1
            Lower_value = Q1 - 1.5*IQR
            Upper_value = Q3 + 1.5*IQR
            sepal_median = np.median(self.df[column])

            for i in self.df[column]:
                if i> Upper_value or i<Lower_value:
                    self.df[column] = self.df[column].replace(i,sepal_median)
            logger.info("Handled ouliers successfully...")
        except Exception as e:
            logger.error(f"Error while handling outliers: {e}")
            raise CustomException("Failed to handle outliers ", e)
        
    def split_data(self):
        try:
            X = self.df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
            Y = self.df["Species"]
            X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size=0.2 , random_state=42)

            logger.info("Data splitted into test and train Sucessfully...")

            joblib.dump(X_train, os.path.join(self.processed_data_path, "X_train.pkl"))
            joblib.dump(X_test, os.path.join(self.processed_data_path, "X_test.pkl"))
            joblib.dump(y_train, os.path.join(self.processed_data_path, "y_train.pkl"))
            joblib.dump(y_test, os.path.join(self.processed_data_path, "y_test.pkl"))

            logger.info("Preprocessed data files saved successfully...")
        except Exception as e:
            logger.error(f"Error while splitting and saving data: {e}")
            raise CustomException("Failed to split the data", e)
        

    def initiate_data_processing(self):
        self.load_data()
        self.handle_outliers("SepalWidthCm")
        self.split_data()

if __name__== "__main__":

    data_pipeline= DataProcessing("artifacts/raw/data.csv")
    data_pipeline.initiate_data_processing()