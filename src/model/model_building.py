import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import yaml
import logging

# Logging configuration
logger = logging.getLogger("model_building")
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

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        learning_rate = params['model_building']['learning_rate']
        n_estimators = params['model_building']['n_estimators']

        parms = {'learning_rate': learning_rate, 'n_estimators': n_estimators}
        return parms
    except FileNotFoundError:
        logger.error("File not Found Error")
        raise
    except Exception as e:
        logger.error("Some error occured!")
        raise

def load_train_data(train_data_path: str) -> pd.DataFrame:
    try:
        train_data = pd.read_csv(train_data_path)
        return train_data
    except FileNotFoundError:
        logger.error(f"Error: The file {train_data_path} was not found.")
        raise
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise

def split_data(train_data: pd.DataFrame) -> tuple:
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    return X_train, y_train

def train_model(parms, X_train, y_train):
    try:
        # Define and train the gradient boosting model
        clf = GradientBoostingClassifier(n_estimators=parms['n_estimators'],
                                         learning_rate=parms['learning_rate'])
        clf.fit(X_train, y_train)
        return clf
    except Exception as e:
        logger.error(f"Error training the model: {e}")
        raise

def save_model(model_path, model):
    try:
        with open(model_path, "wb") as file:
            pickle.dump(model, file)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving the model: {e}")
        raise

def main():
    try:
        # Load parameters
        parms = load_params('params.yaml')

        # Load training data
        train_data = load_train_data('./data/processed/train_tfidf.csv')

        # Split data into features and target
        X_train, y_train = split_data(train_data=train_data)

        # Train the model
        model = train_model(parms, X_train, y_train)

        # Save the trained model
        save_model("models/model.pkl", model)

    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}")

if __name__ == "__main__":
    main()
