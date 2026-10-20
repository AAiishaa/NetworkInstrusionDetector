# This file holds: Dataset paths, model paths and shared settings

# to build file and folder paths
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
#.parents[2] moves up 3 directory levels from the files location

#defining folder paths


# main data folder
data_dir = project_root / "data"

# orignal data sets
raw_data_dir = data_dir / "raw"

#saved machine learning models
model_dir = project_root / "models"

# evaluation results and report
reports_dir = project_root / "reports"

# images such as confusion matrices
figures_dir = reports_dir / "figures"

# pointing to the dataset files

train_data_path = raw_data_dir / "KDDTrain+.txt"
test_data_path = raw_data_dir / "KDDTest+.txt"



# defining where the models will be saved

# train model selected after comapring Logistic Regression and Random Forest
model_path = model_dir / "best_model.joblib"
preprocessor_path = model_dir / "preprocessor.joblib"

#info about the model such as name, features and training details
model_metadata_path = model_dir / "model_metadata.json"



# deifning evaluation path

#summary metrics such as accuracy, precision, recall, F1-score 
metics_path = reports_dir / "metrics.json"
classification_report_path = reports_dir / "classification_report.json"
# visual breakdown of correct and incorrect predictions.
confusion_matrix_path = figures_dir / "confusion_matrix.png"


#include dataset and experiment settings
# to keep important settings in one place
# makes project easier to maintain, only change setting once instead of upadating acroos multiple files

random_state = 42

#represasents 20% of dataset when creating a split
test_size = 0.2

#desribing NSL KDD dataset structure

expected_feature_count = 41
#label colum describing connection class - whats bieng learnt
# difficulty describes difficulty of classfiying the exampple (part of the metadata)
label_column= "label"
difficulty_column = "difficulty"

# categorical features(which will be numerically prepreasented)

categorical_features = [
    "protocol_type",
    "service",
    "flag"
]

# deifning binary classification
# model predicts one of two outcomes
normal_label = "normal"
normal_target = 0
attack_target = 1



