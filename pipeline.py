import sys
import os
import mlflow

import pandas as pd
from evaluate import evaluate_model
from model import save_model, validate_model

from data_validations import (
    Validationlogger,
    DataValidation,
    INPUT_FILE_PATH,
    OUTPUT_FILE_PATH,
    LOG_FILE_PATH,
    validation_rules
)

from preprocessing import (
    prepare_data,
    train_model
)

from model import save_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "attrition_pipeline.pkl"
)



def run_pipeline():

    print("=" * 60)
    print("EMPLOYEE ATTRITION MLOPS PIPELINE")
    print("=" * 60)

    # --------------------------------------------------
    # STEP 1: VALIDATE ORIGINAL DATA
    # --------------------------------------------------

    print("\n[1] Running Data Validation...")

    logger = Validationlogger(LOG_FILE_PATH)

    validator = DataValidation(
        INPUT_FILE_PATH,
        OUTPUT_FILE_PATH,
        validation_rules,
        logger
    )

    validation_ok = validator.validate_data()

    if not validation_ok:

        print("Original data contains invalid records.")
        print("Invalid records have been quarantined.")
        print("Cleaned dataset has been created.")

        # --------------------------------------------------
        # STEP 1A: RE-VALIDATE CLEANED DATA
        # --------------------------------------------------

        print("\n[1A] Re-validating cleaned dataset...")

        cleaned_validator = DataValidation(
            OUTPUT_FILE_PATH,
            OUTPUT_FILE_PATH,
            validation_rules,
            logger
        )

        cleaned_validation_ok = cleaned_validator.validate_data()

        if not cleaned_validation_ok:

            print("Cleaned Dataset Validation: FAILED")
            print("Pipeline stopped.")

            return False

        print("Cleaned Dataset Validation: PASSED")

    else:

        print("Validation Gate: PASSED")

    # --------------------------------------------------
    # STEP 2: PREPARE DATA
    # --------------------------------------------------

    print("\n[2] Preparing Data...")

    X_train, X_test, y_train, y_test = prepare_data()

    print("Data preparation completed.")

    # --------------------------------------------------
    # STEP 3: TRAIN + VALIDATE MODEL
    # --------------------------------------------------

    print("\n[3] Training Model...")

    model_pipeline = train_model(
        X_train,
        y_train
    )

    if model_pipeline is None:
        print("Model Training: FAILED")
        print("Pipeline stopped.")
        return False

    # --------------------------------------------------
    # STEP 4: EVALUATE MODEL
    # --------------------------------------------------

    print("\n[4] Evaluating Model...")

    accuracy, precision, recall, f1, cm = evaluate_model(
        model_pipeline,
        X_test,
        y_test
    )
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    model_ok = validate_model(
    accuracy,
    recall,
    f1
)

    model_ok = validate_model(
        accuracy,
        recall,
        f1
    )

    if not model_ok:
        print("Model Evaluation Gate: FAILED")
        print("Pipeline stopped.")
        return False

    print("Model Evaluation Gate: PASSED")

    # --------------------------------------------------
    # STEP 5: SAVE APPROVED MODEL
    # --------------------------------------------------

    print("\n[5] Saving Approved Model...")

    save_model(
        model_pipeline,
        MODEL_PATH
    )

    print("Model artifact saved successfully.")

    # --------------------------------------------------
    # STEP 6: RUN PREDICTION
    # --------------------------------------------------

    print("\n[6] Running Prediction...")

    new_employee = pd.DataFrame([{
        "Age": 35,
        "Department": "Sales",
        "Education": "Medical",
        "JobRole": "Sales Executive",
        "JobSatisfaction": 3,
        "MonthlyIncome": 5000,
        "OverTime": "No",
        "YearsAtCompany": 5
    }])

    
    prediction = model_pipeline.predict(new_employee)[0]
        

    print("Prediction completed successfully.")

    # --------------------------------------------------
    # PIPELINE SUCCESS
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return True


if __name__ == "__main__":

    mlflow.set_experiment("Employee_Attrition")

    with mlflow.start_run():

        mlflow.log_param(
            "model_type",
            "LogisticRegression"
        )

        mlflow.log_param(
            "max_iter",
            1000
        )

        mlflow.log_param(
            "class_weight",
            "balanced"
        )

        success = run_pipeline()

    sys.exit(0 if success else 1)
    