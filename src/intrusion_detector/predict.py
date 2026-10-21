# Load the saved model and make predictions

import joblib
import pandas as pd

from .config import model_path, reports_dir
from .data import load_testing_data_without_satan
from .preprocessing import split_features_and_target


def predict():

    # load testing dataset

    data = load_testing_data_without_satan()

    X, y = split_features_and_target(data)

    # load saved pipline
    # the pipline includes preprocessing anf the radom forest classifier
    model = joblib.load(model_path)

    # make predicitions on the testing data
    predictions = model.predict(X)


    # Return the actual and predicted labels for evaluation
    return y, predictions

if __name__ == "__main__":
    actual, predictions = predict()

    print("Predictions completed successfully.")
    print(f"Total predictions: {len(predictions)}")

