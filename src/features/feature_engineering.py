import numpy as np
import pandas as pd
import yaml
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

# Logging configuration
logger = logging.getLogger("feature_engineering")
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

def load_params(params_path: str) -> int:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        max_features = params['feature_engineering']['max_features']
        logger.debug("Max features retrieved from params.yaml")
        return max_features
    except FileNotFoundError as e:
        logger.error(f"File not found: {params_path}. Please check the path.")
        raise
    except KeyError as e:
        logger.error(f"KeyError: Missing expected key in {params_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise

def load_data(train_data_path: str, test_data_path: str) -> tuple:
    try:
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        logger.info("Train and test data loaded successfully")
        return train_data, test_data
    except FileNotFoundError as e:
        logger.error(f"File not found: {e.filename}. Please check the path.")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def transform_data(max_features: int, train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
    """Apply TfIdf to the data."""
    try:
        logger.info("Starting data transformation using Bag of Words")

        # Ensure 'content' column exists
        if 'content' not in train_data.columns or 'content' not in test_data.columns:
            raise KeyError("The 'content' column is missing in train or test data.")

        # Fill NaN values with empty strings
        train_data['content'] = train_data['content'].fillna("")
        test_data['content'] = test_data['content'].fillna("")

        # Extract features and labels
        X_train = train_data['content']
        y_train = train_data['sentiment']

        X_test = test_data['content']
        y_test = test_data['sentiment']

        # Initialize CountVectorizer
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        # Convert sparse matrix to DataFrame
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train.values  # Ensure alignment

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test.values  # Ensure alignment

        logger.info("Data transformation completed successfully")
        return train_df, test_df

    except KeyError as e:
        logger.error(f"KeyError: Missing column in the data: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during data transformation: {e}")
        raise

def save_transformed_data(data_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_file = os.path.join(data_path, "train_tfidf.csv")
        test_file = os.path.join(data_path, "test_tfidf.csv")
        
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        logger.info(f"Transformed data saved successfully in {data_path}")
    except Exception as e:
        logger.error(f"Error saving transformed data: {e}")
        raise

def main():
    try:
        logger.info("Starting main data processing pipeline")
        max_features = load_params('params.yaml')
        train_data, test_data = load_data('./data/interim/train_processed.csv',
                                          './data/interim/test_processed.csv')
        train_df, test_df = transform_data(max_features, train_data, test_data)
        data_path = os.path.join("data", "processed")
        save_transformed_data(data_path, train_df, test_df)
        logger.info("Data processing pipeline completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during the main process: {e}")
        raise

if __name__ == "__main__":
    main()