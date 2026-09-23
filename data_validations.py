import pandas as pd
import os
import logging
from collections import defaultdict
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILE_PATH = os.path.join(BASE_DIR, "logs", "validation.log")
INPUT_FILE_PATH = os.path.join(BASE_DIR, "data", "HR-Employee-Attrition.csv")
OUTPUT_FILE_PATH = os.path.join(BASE_DIR, "data", "HR-Employee-Attrition_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "attrition_model.pkl")



validation_rules = {
    "Age": {"type": int, "min": 18, "max": 65},
    "Department": {"type": str, "allowed_values": ["Sales", "Research & Development", "Human Resources"]},
    "Education": {"type": str, "allowed_values": ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]},
    "EmployeeId": {"type": int, "min": 1},
    "JobRole": {"type": str, "allowed_values": ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"]},
    "JobSatisfaction": {"type": int, "min": 1, "max": 4},
    "MonthlyIncome": {"type": int, "min": 1000, "max": 20000},
    "OverTime": {"type": str, "allowed_values": ["Yes", "No"]},
    "YearsAtCompany": {"type": int, "min": 0, "max": 40}
}

log_dir = os.path.dirname(LOG_FILE_PATH)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)


class Validationlogger:
    def __init__(self, log_file_path):
        self.logger = logging.getLogger("ValidationLogger")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            log_dir = os.path.dirname(log_file_path)

            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(
                log_file_path,
                mode="a",
                encoding="utf-8"
            )

            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )

            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            self.logger.info("Logger initialized successfully")
            self.logger.info("Log file path: %s", log_file_path)

    def log_dataframe_profile(self, df, dataset_name="dataset"):
        self.logger.info("%s columns: %s", dataset_name, list(df.columns))
        self.logger.info("%s shape: rows=%d, cols=%d", dataset_name, df.shape[0], df.shape[1])

    def log_dataframe_nulls(self, df, dataset_name="dataset"):
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        self.logger.info("%s total null values: %d", dataset_name, total_nulls)
        if total_nulls > 0:
            self.logger.info("%s null values by column:\n%s", dataset_name, null_counts[null_counts > 0])

    def log_dataframe_duplicates(self, df, dataset_name="dataset"):
        duplicate_count = df.duplicated().sum()
        self.logger.info("%s total duplicate rows: %d", dataset_name, duplicate_count)
        if duplicate_count > 0:
            self.logger.info("%s duplicate rows:\n%s", dataset_name, df[df.duplicated()])


class DataValidation:
    def __init__(self, input_file_path, output_file_path, validation_rules, logger: Validationlogger):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.validation_rules = validation_rules
        self.validation_logger = logger

    def _check_type(self, series, expected_type):
        if expected_type is int:
            return pd.api.types.is_integer_dtype(series) or pd.api.types.is_numeric_dtype(series)
        if expected_type is str:
            return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
        return True

    def validate_data(self):
        try:
            df = pd.read_csv(self.input_file_path)
        except Exception as e:
            self.validation_logger.logger.error("Failed to read input file %s: %s", self.input_file_path, str(e))
            return False

        self.validation_logger.log_dataframe_profile(df, "Input Dataset")
        self.validation_logger.log_dataframe_nulls(df, "Input Dataset")
        self.validation_logger.log_dataframe_duplicates(df, "Input Dataset")

        validation_errors = []
        # Track invalid rows and reasons per row
        invalid_mask = pd.Series(False, index=df.index)
        invalid_reasons = defaultdict(list)

        for column, rules in self.validation_rules.items():
            if column not in df.columns:
                self.validation_logger.logger.warning("Column '%s' not found in dataset", column)
                continue

            series = df[column]

            if "type" in rules:
                expected_type = rules["type"]
                # Per-row type checks: mark rows that can't be coerced to expected type
                if expected_type is int:
                    coerced = pd.to_numeric(series, errors="coerce")
                    non_numeric = coerced.isna()
                    if non_numeric.any():
                        invalid_mask |= non_numeric
                        for idx in df[non_numeric].index:
                            invalid_reasons[idx].append(f"Non-integer value in '{column}'")
                        validation_errors.append((column, f"Incorrect type for some rows. Expected: {expected_type}, Found: {series.dtype}"))
                elif expected_type is str:
                    non_str = ~series.apply(lambda x: isinstance(x, str))
                    non_str = non_str.fillna(True)
                    if non_str.any():
                        invalid_mask |= non_str
                        for idx in df[non_str].index:
                            invalid_reasons[idx].append(f"Non-string value in '{column}'")
                        validation_errors.append((column, f"Incorrect type for some rows. Expected: {expected_type}, Found: {series.dtype}"))

            if "min" in rules:
                min_value = rules["min"]
                try:
                    numeric_series = pd.to_numeric(series, errors="coerce")
                    invalid_min_mask = numeric_series < min_value
                    invalid_min_mask = invalid_min_mask.fillna(False)
                    if invalid_min_mask.any():
                        invalid_mask |= invalid_min_mask
                        for idx in df[invalid_min_mask].index:
                            invalid_reasons[idx].append(f"Value below min ({min_value}) in '{column}'")
                        validation_errors.append((column, f"Values below minimum ({min_value}): {invalid_min_mask.sum()} rows"))
                except Exception:
                    pass

            if "max" in rules:
                max_value = rules["max"]
                try:
                    numeric_series = pd.to_numeric(series, errors="coerce")
                    invalid_max_mask = numeric_series > max_value
                    invalid_max_mask = invalid_max_mask.fillna(False)
                    if invalid_max_mask.any():
                        invalid_mask |= invalid_max_mask
                        for idx in df[invalid_max_mask].index:
                            invalid_reasons[idx].append(f"Value above max ({max_value}) in '{column}'")
                        validation_errors.append((column, f"Values above maximum ({max_value}): {invalid_max_mask.sum()} rows"))
                except Exception:
                    pass

            if "allowed_values" in rules:
                allowed_values = rules["allowed_values"]
                try:
                    invalid_values_mask = ~series.isin(allowed_values)
                    invalid_values_mask = invalid_values_mask.fillna(True)
                    if invalid_values_mask.any():
                        invalid_mask |= invalid_values_mask
                        for idx in df[invalid_values_mask].index:
                            invalid_reasons[idx].append(f"Value not in allowed set for '{column}'")
                        validation_errors.append((column, f"Values outside allowed set ({allowed_values}): {invalid_values_mask.sum()} rows"))
                except Exception:
                    pass

        # Log column-level validation errors
        if validation_errors:
            for column, error in validation_errors:
                self.validation_logger.logger.warning("Validation error in column '%s': %s", column, error)

        # If any invalid rows found, save them to a quarantine file with reasons
        quarantine_path = os.path.join(os.path.dirname(self.output_file_path), "quarantine.csv")
        if invalid_mask.any():
            df_quarantine = df[invalid_mask].copy()
            df_quarantine["validation_errors"] = [
                "; ".join(invalid_reasons[idx]) for idx in df_quarantine.index
            ]
            os.makedirs(os.path.dirname(quarantine_path), exist_ok=True)
            df_quarantine.to_csv(quarantine_path, index=False)
            self.validation_logger.logger.info(
                "Quarantined %d invalid rows to: %s",
                df_quarantine.shape[0],
                quarantine_path,
            )
            df_cleaned = df[~invalid_mask].copy()

            os.makedirs(os.path.dirname(self.output_file_path), exist_ok=True)
            
            df_cleaned.to_csv(self.output_file_path, index=False)
            self.validation_logger.logger.info(
                "Cleaned dataset shape: rows=%d, cols=%d",
                df_cleaned.shape[0],
                df_cleaned.shape[1]
                )
            self.validation_logger.logger.info(
                "Valid records saved to cleaned dataset: %s",
                self.output_file_path,
            )

            if validation_errors or invalid_mask.any():
                self.validation_logger.logger.error("Validation gate: FAILED")
                return False

        # Save cleaned data (drop exact duplicate rows)
        df_cleaned = df.drop_duplicates()
        os.makedirs(os.path.dirname(self.output_file_path), exist_ok=True)
        df_cleaned.to_csv(self.output_file_path, index=False)
        self.validation_logger.logger.info("Cleaned dataset saved to: %s", self.output_file_path)
        self.validation_logger.logger.info("Validation gate: PASSED")
        return True


if __name__ == "__main__":
    logger = Validationlogger(LOG_FILE_PATH)
    validator = DataValidation(INPUT_FILE_PATH, OUTPUT_FILE_PATH, validation_rules, logger)
    success = validator.validate_data()
    print("Validation success:", success)