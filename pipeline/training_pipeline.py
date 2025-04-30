from src.data_processing import DataProcessing
from src.model_training import ModelTraining


if __name__== "__main__":

    data_pipeline= DataProcessing("artifacts/raw/data.csv")
    data_pipeline.initiate_data_processing()

    training_pipeline = ModelTraining()
    training_pipeline.run()