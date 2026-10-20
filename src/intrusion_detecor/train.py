# Train random forest model using KDDTrain+.txt

# saves model and preprocessor to files to use later
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .config import (
    model_dir,
    model_path,
    preprocessor_path,
    random_state,
)

from .data import load_training_data
from .preprocessing import build_preprocessor, split_features_and_target


def train_model():
    # Load the training dataset
    data = load_training_data()

    # x contains the 41 input features
    # y contains the correct label 0 or 1
    X, y = split_features_and_target(data)

  

    # Create a pipeline
    # A pipeline combines multiple machine learning steps into one workflow.
    # Step 1: "preprocessing" converts categorical features into numerical values.
    # Step 2: "classifier" uses the Random Forest to learn patterns in the data.
    # The steps run in order, so we do not need to preprocess X separately.
    model = Pipeline([
        ("preprocessing", build_preprocessor(X.columns)),
        ("classifier", RandomForestClassifier(
            n_estimators=100,       # Build 100 decision trees
            random_state=random_state,  # Make training reproducible
            class_weight="balanced",   # Adjust for differences in class frequency
            n_jobs=-1,              # Use all available CPU cores
        ))
    ])

    # Train the pipeline
    # fit() first learns the preprocessing rules, transforms X,
    # and then trains the Random Forest using X and the correct labels y.
    model.fit(X, y)

    # Create the models directory if it does not exist
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save the complete pipeline
    # This includes both the fitted preprocessor and the trained Random Forest.
    joblib.dump(model, model_path)

    # Save the preprocessor separately in case we need it independently
    joblib.dump(
        model.named_steps["preprocessing"],
        preprocessor_path,
    )

    print("Model trained and saved successfully.")
    print(f"Model pipeline saved to: {model_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")


if __name__ == "__main__":
    train_model()