# retrain.py - Refreshes the production model with new training data

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .config import model_path, random_state, model_dir
from .data import load_training_data
from .preprocessing import build_preprocessor, split_features_and_target


def retrain_model():
    print("Loading training data...")
    data = load_training_data()
    X, y = split_features_and_target(data)

    print("Building and fitting new model pipeline...")
    # Recreate the pipeline using your exact setup from train.py
    model = Pipeline([
        ("preprocessing", build_preprocessor(X.columns)),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    # Fit model on the data
    model.fit(X, y)

    # Automatically create models folder if it was deleted
    model_dir.mkdir(parents=True, exist_ok=True)

    # Overwrite the saved model
    joblib.dump(model, model_path)
    print(f"Model retrained and updated successfully at: {model_path}")


if __name__ == "__main__":
    retrain_model()