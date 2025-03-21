import numpy as np
import pandas as pd
import pickle
import json
import logging
from sklearn.metrics import (accuracy_score,
                             recall_score, precision_score, roc_auc_score)

# Logging configuration
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# create File handler
file_handler = logging.FileHandler('errors.log')
file_handler.setLevel("ERROR")

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_test_data(path: str) -> pd.DataFrame:
    try:
        test_data = pd.read_csv(path)
        return test_data
    except FileNotFoundError:
        logger.error(f"Error: The file {path} was not found.")
        raise
    except Exception as e:
        logger.error(f"Error loading test data from {path}: {e}")
        raise

def split_data(test_data: pd.DataFrame) -> tuple:
    try:
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        return X_test, y_test
    except KeyError as e:
        logger.error(f"KeyError: Missing expected column in test data: {e}")
        raise
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise

def load_model(path: str):
    try:
        with open(path, "rb") as file:
            clf = pickle.load(file)
        return clf
    except FileNotFoundError:
        logger.error(f"Error: The model file {path} was not found.")
        raise
    except Exception as e:
        logger.error(f"Error loading model from {path}: {e}")
        raise

def make_prediction(clf, X_test):
    try:
        # Make prediction
        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)[:, 1]
        return y_pred, y_pred_prob
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise

def calculate_eval(y_test, y_pred, y_pred_prob):
    try:
        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)

        metrics_dict = {
            'accuracy_score': acc,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

        return metrics_dict
    except Exception as e:
        logger.error(f"Error calculating evaluation metrics: {e}")
        raise

def save_metrics(path, metrics_dict):
    try:
        with open(path, 'w') as file:
            json.dump(metrics_dict, file, indent=4)
        logger.info(f"Metrics saved to {path}")
    except Exception as e:
        logger.error(f"Error saving metrics to {path}: {e}")
        raise

def main():
    try:
        # Load test data
        test_data = load_test_data('./data/processed/test_bow.csv')

        # Split the test data into features and labels
        X_test, y_test = split_data(test_data)

        # Load the trained model
        model = load_model("models/model.pkl")

        # Make predictions
        y_pred, y_pred_prob = make_prediction(model, X_test)

        # Calculate evaluation metrics
        metrics_dict = calculate_eval(y_test, y_pred, y_pred_prob)

        # Save the metrics to a JSON file
        save_metrics("reports/metrics.json", metrics_dict)

    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}")

if __name__ == "__main__":
    main()
