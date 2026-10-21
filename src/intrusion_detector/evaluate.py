# Calculate and save evaluation metrics

# savesm reuslts into json files
import json
import matplotlib.pyplot as plt

# create confsion matrix heatmap
import seaborn as sns


from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# paths where the results wi;;
from .config import(
    reports_dir,
    metrics_path,
    classification_report_path,
    confusion_matrix_path
)

from .predict import predict


def evaluate_model():
    # get the actual labels and the model predictions from predict .py

    actual, predictions = predict()

    # calculate main evaluation metrics
    # zero_division=0, if the model makes no posotive predictions

    metrics = {

        "accuracy" : accuracy_score(actual, predictions),
        "precision" : precision_score(actual, predictions, zero_division=0),
        "recall": recall_score(actual, predictions, zero_division=0),
        "f1_score": f1_score(actual, predictions, zero_division=0),

    }



    # calculate confusion matrix

    matrix = confusion_matrix(actual, predictions, labels = [0, 1])
    
    # extracts the followng values from the matrix
    tn, fp, fn, tp = matrix.ravel()

    # Counts correct and incorrect predictions:
    # tn - normal traffic correctly classified.
    # fp — normal traffic incorrectly flagged as an attack.
    # fn — an attack incorrectly classified as normal.
    # tp — an attack correctly detected.

    # calculate false postive rate

    metrics["false_positive_rate"] = (
    fp / (fp + tn) if (fp + tn) > 0 else 0
    )


    # create report folder

    reports_dir.mkdir(parents=True, exists_ok =True)


    # save the metrics

    with open(metrics_path, "w") as file:
        json.dump(
            {key: float(value) for key, value in metrics.items()},
            file,
            indent=4,
        )


    report = classification_report(
        actual,
        predictions,
        labels = [0, 1],
        target_names=["normal", "attack"],
        output_dict=True, # retund the report as a dictionary
        zero_division=0,
    )

    with open(classification_report_path, "w") as file:
        json.dump(report, file, indent=4)


        # report saved as classification_report.json


    # create seaborn heatmap

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix, # confusion matrix data
        annot=True, # display numbers inside each cell
        fmt = "d", # display numbers as integers
        cmap = "Blues",
        xticklabels=["Normal", "Attack"], # labels the predicted classes
        yticklabels=["Normal", "Attack"], # labels the actual classes

    )

    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()


    # Save the confusion matrix image in the reports directory
    plt.savefig(confusion_matrix_path)
    plt.close()

      # Display the evaluation results in the terminal
    print("Model evaluation completed successfully.")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Display where the results were saved
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Classification report saved to: {classification_report_path}")
    print(f"Confusion matrix saved to: {confusion_matrix_path}")



    

    




