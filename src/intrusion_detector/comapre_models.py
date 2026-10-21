
# Compare the original and retrained models on unseen satan attacks

import joblib
from sklearn.metrics import recall_score

from .config import model_path
from .data import load_testing_data
from .preprocessing import split_features_and_target


def compare_models():
    # Load satan attacks from the test dataset
    data = load_testing_data
    X, y = split_features_and_target(data)

    # Load the original model
    original_model = joblib.load(model_path)

    # Load the retrained model
    retrained_model_path = model_path.with_name(
        f"{model_path.stem}_retrained{model_path.suffix}"
    )
    retrained_model = joblib.load(retrained_model_path)

    # Make predictions using both models
    original_predictions = original_model.predict(X)
    retrained_predictions = retrained_model.predict(X)

    # Calculate recall for both models
    original_recall = recall_score(
        y, original_predictions, zero_division=0
    )
    retrained_recall = recall_score(
        y, retrained_predictions, zero_division=0
    )

    # Display the comparison
    difference = retrained_recall - original_recall

    print("Satan Attack Detection Comparison")
    print("--------------------------------")
    print(f"Original model recall:  {original_recall:.2%}")
    print(f"Retrained model recall: {retrained_recall:.2%}")
    print(f"Recall difference:      {difference:+.2%}")


if __name__ == "__main__":
    compare_models()
