
# Run the isolated new-attack retraining experiment
# using the full training dataset including satan attacks

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from .config import model_path, random_state
from .data import load_training_data, load_testing_data
from .preprocessing import build_preprocessor, split_features_and_target


def test_original_on_new_attacks():
    # Test the original model on unseen satan attacks
    satan_data = load_testing_data
    X_satan, y_satan = split_features_and_target(satan_data)

    original_model = joblib.load(model_path)
    predictions = original_model.predict(X_satan)

    print("Original model performance on unseen satan attacks:")
    print(classification_report(y_satan, predictions, zero_division=0))


def retrain():
    # Load full training dataset, including satan attacks
    data = load_training_data()

    X, y = split_features_and_target(data)

    # Create and train a new model
    model = Pipeline([
        ("preprocessing", build_preprocessor(X.columns)),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    model.fit(X, y)

    # Save separately from the original model
    retrained_model_path = model_path.with_name(
        f"{model_path.stem}_retrained{model_path.suffix}"
    )
    joblib.dump(model, retrained_model_path)

    print(f"Retrained model saved to: {retrained_model_path}")


def test_new_model_with_satan():
    # Test the retrained model on unseen satan attacks
    satan_data = load_testing_data
    X_satan, y_satan = split_features_and_target(satan_data)

    retrained_model_path = model_path.with_name(
        f"{model_path.stem}_retrained{model_path.suffix}"
    )
    retrained_model = joblib.load(retrained_model_path)
    predictions = retrained_model.predict(X_satan)

    print("Retrained model performance on unseen satan attacks:")
    print(classification_report(y_satan, predictions, zero_division=0))


if __name__ == "__main__":
    test_original_on_new_attacks()
    retrain()
    test_new_model_with_satan()
