
# Load the NSL-KDD files and validate their columns
# reads the datasets
# check the data has 43 columns
# clean up text labels
# check for missing values and invalid data
# returns the cleaned data for the next stage

# IMPORTANT this file checks the structure and basic data quality

from pathlib import Path

# to load, organise and analyse data
import pandas as pd

# import settings from the config.py
from .config import (
    categorical_features,
    difficulty_column,
    expected_feature_count,
    label_column,
    test_data_path,
    train_data_path,
)

# NSL-KDD contains 41 input features, followed by the label and difficulty (43 in total)
# the text files contain data without header rows
columns_names = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    label_column,  # = "label"
    difficulty_column,  # = "difficulty"
]


# loading dataset function
def _load_dataset(file_path: Path, dataset_name: str) -> pd.DataFrame:

    # check if the file exists using the .is_file() method
    if not file_path.is_file():
        raise FileNotFoundError(
            f"{dataset_name} dataset not found at: {file_path}"
        )

    # reading the file
    # pandas reads the file into memory
    # header=None tells pandas the first row contains data, not headings
    # skipinitialspace skips spaces immediately after delimiters
    data = pd.read_csv(
        file_path,
        header=None,
        skipinitialspace=True,
    )

    # validating number of columns
    expected_column_count = expected_feature_count + 2

    # data.shape[1] is the number of columns
    if data.shape[1] != expected_column_count:
        raise ValueError(
            f"{dataset_name} dataset has {data.shape[1]} columns; "
            f"expected {expected_column_count} "
            f"({expected_feature_count} features, label and difficulty)."
        )

    # assigning column names
    # once the number of columns has been checked,
    # assign the 43 names to the table
    data.columns = columns_names

    # cleaning text values by removing whitespace
    # prevent the model from treating values differently because of accidental spaces
    text_columns = data.select_dtypes(include=["object"]).columns

    for column in text_columns:
        data[column] = data[column].str.strip()

    # standardising attack labels
    # remove trailing periods and convert labels to lowercase
    data[label_column] = (
        data[label_column]
        .str.lower()
        .str.rstrip(".")
    )

    # checking labels are valid
    # is there a missing value or an empty string?
    if data[label_column].isna().any() or (data[label_column] == "").any():
        raise ValueError(
            f"{dataset_name} dataset contains missing or empty labels."
        )

    # validating the difficulty column
    data[difficulty_column] = pd.to_numeric(
        data[difficulty_column],
        errors="raise",
    )

    # check for missing values
    missing_counts = data.isna().sum()
    missing_columns = missing_counts[missing_counts > 0]

    if not missing_columns.empty:
        details = missing_columns.to_dict()
        raise ValueError(
            f"{dataset_name} dataset contains missing values: {details}"
        )

    
    return data


# 1. Load all training data
def load_training_data():
    return _load_dataset(train_data_path, "Training")


# 2. Load all test data
def load_testing_data():
    return _load_dataset(test_data_path, "Testing")


# 3. Load training data WITHOUT satan
def load_training_data_without_satan():
    data = load_training_data()
    return data[data[label_column] != "satan"].copy()


# 4. Load test data WITHOUT satan
def load_testing_data_without_satan():
    data = load_testing_data()
    return data[data[label_column] != "satan"].copy()