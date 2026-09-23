
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


import pandas as pd
from sklearn.compose import ColumnTransformer

from model import create_model_pipeline,  save_model

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLEANED_FILE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "HR-Employee-Attrition_cleaned.csv"
)


# ---------------------------------------------------------
# Column definitions
# ---------------------------------------------------------

numeric_columns = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "JobSatisfaction"
]

categorical_columns = [
    "Education",
    "Department",
    "OverTime",
    "JobRole"
]


# ---------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_columns),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
    ],
    remainder="passthrough"
)


def prepare_data():

    df = pd.read_csv(CLEANED_FILE_PATH)     
    

    # Separate features and target
    X = df.drop(columns=["Attrition", "EmployeeId"])
    y = df["Attrition"]

    print("Original shape:", X.shape)

    # Split raw data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    print("\nRaw data split completed successfully.")

    return X_train, X_test, y_train, y_test


def train_model(
    X_train,
    y_train
):

    # Create preprocessing + model pipeline
    model_pipeline = create_model_pipeline(preprocessor)

    # Train
    model_pipeline.fit(X_train, y_train)

    print(
        "Preprocessing + model training "
        "completed successfully."
    )

    return model_pipeline
if __name__ == "__main__":

    X_train, X_test, y_train, y_test = prepare_data()

    model_pipeline = train_model(
        X_train,
        y_train
    )

    MODEL_PATH = (
        r"C:\Users\tejch\OneDrive\Desktop\Python_Practice"
        r"\Emp_attrition\models\attrition_pipeline.pkl"
    )

    save_model(
        model_pipeline,
        MODEL_PATH
    )