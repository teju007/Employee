from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
from sklearn.pipeline import Pipeline
import mlflow



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "attrition_model.pkl"
)


def train_model(X_train, y_train):

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Model training completed successfully.")

    return model

def evaluate_model(model, X_test, y_test):
     predictions = model.predict(X_test)
     accuracy = accuracy_score(y_test, predictions) 
     print("Model Accuracy:", accuracy)
     return accuracy


def validate_model(
    accuracy,
    recall,
    f1,
    minimum_accuracy=0.70,
    minimum_recall=0.50,
    minimum_f1=0.40
):

    if (
        accuracy >= minimum_accuracy
        and recall >= minimum_recall
        and f1 >= minimum_f1
    ):

        print("Model Validation Gate: PASSED")

        print(
            f"Accuracy {accuracy:.2%} >= "
            f"{minimum_accuracy:.2%}"
        )

        print(
            f"Recall {recall:.2%} >= "
            f"{minimum_recall:.2%}"
        )

        print(
            f"F1 Score {f1:.2%} >= "
            f"{minimum_f1:.2%}"
        )

        return True

    print("Model Validation Gate: FAILED")

    print(
        f"Accuracy {accuracy:.2%} "
        f"(minimum {minimum_accuracy:.2%})"
    )

    print(
        f"Recall {recall:.2%} "
        f"(minimum {minimum_recall:.2%})"
    )

    print(
        f"F1 Score {f1:.2%} "
        f"(minimum {minimum_f1:.2%})"
    )

    return False

def save_model(model, model_path=MODEL_PATH):
    model_dir = os.path.dirname(model_path)

    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, model_path)

    print(f"Model saved successfully to: {model_path}")


def create_model_pipeline(preprocessor):

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
         class_weight="balanced"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline




