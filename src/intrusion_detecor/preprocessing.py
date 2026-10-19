# prepares network traffic data for the ML
# Separate labels, encode categorical features and build the preprocessing pipeline
# both models will use the same preprocessing pl for fairness


import pandas as pd
# appl different tranformations to different columns
from sklearn.compose import ColumnTransformer

# onehotencoder converts categories into numerical features
from sklearn.preprocessing import OneHotEncoder

from .config import(
    attack_target,
    categorical_features,
    difficulty_column,
    label_column,
    normal_label,
    normal_target
)

# seperate input featueres from the target label
# X: the info the model recives as inout
# Y: the correct answer the model learns to predict

def split_features_and_target(data):
    
    X = data.drop(columns=[label_column, difficulty_column])

    # .astype("int64") converts True to 1 and False to 0
    y = (data[label_column] != normal_label).astype("int64")

    return X , y


def build_preprocessor(feature_columns):

    # check that the required categorial features exist

    missing = set(categorical_features) - set(feature_columns)

    if missing:
        raise ValueError(f"Missing categorical features: {sorted(missing)}")


# ColumnTransformer combines transformations for selected columns into one object
# Here, we tell it to apply one-hot encoding to the categorical features
    return ColumnTransformer(

        # list transformation to apply
        transformers=[
            (
                "categorical", # name for the trnaformation
                OneHotEncoder(handle_unknown="ignore"), # if a new category appears dont crash
                categorical_features, # specifies which columns to encode
            ),
        ],
        remainder="passthrough", # leave the rest of the columns the same instead of removing them
    )

        




