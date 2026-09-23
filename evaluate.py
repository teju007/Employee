from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        pos_label="Yes"
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="Yes"
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="Yes"
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["No", "Yes"]
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("Model Accuracy:", accuracy)
    print("Model Precision:", precision)
    print("Model Recall:", recall)
    print("Model F1 Score:", f1)

    return accuracy, precision, recall, f1, cm

    model_ok = validate_model(
    accuracy,
    recall,
    f1
)

    if not model_ok:
        print("Model Evaluation Gate: FAILED")
        return False, accuracy

    print("Model Evaluation Gate: PASSED")

    return True, accuracy

if __name__ == "__main__":

    from preprocessing import prepare_data

    X_train, X_test, y_train, y_test = prepare_data()

    import joblib

    MODEL_PATH = (
        r"C:\Users\tejch\OneDrive\Desktop\Python_Practice"
        r"\Emp_attrition\models\attrition_pipeline.pkl"
    )

    model = joblib.load(MODEL_PATH)

    evaluate_model(
        model,
        X_test,
        y_test
    )
    