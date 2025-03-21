import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
import os

# logging configure
logger = logging.getLogger('data_ingestion')
logger.setLevel("DEBUG")

# create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

# create File handler
file_handler = logging.FileHandler('errors.log')
file_handler.setLevel("ERROR")

# create a format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> float:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        test_size = params['data_ingestion']['test_size']
        logger.debug('test size retrieved')
        return test_size
    except FileNotFoundError as e:
        logger.error("File Not found")
        raise
    except yaml.YAMLError as e:
        logger.error("yaml error")
        raise
    except Exception as e:
        logger.error("Some error occured")
        print(e)
        raise


def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        print(f"Error: Unexpected error occurred while reading the data from {url}.")
        print(e)
        raise


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Drop unnecessary columns
        df.drop(columns=['tweet_id'], errors='ignore', inplace=True)

        # Filter out rows with only 'happiness' or 'sadness' sentiments
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()

        # Map sentiments to numeric values (1 for happiness, 0 for sadness)
        final_df['sentiment'] = final_df['sentiment'].map({'happiness': 1, 'sadness': 0})

        # Remove rows with missing values
        final_df.dropna(inplace=True)
        
        # Ensure the 'sentiment' column is of integer type
        final_df['sentiment'] = final_df['sentiment'].astype(int)

        return final_df

    except KeyError as e:
        print(f"Error: Missing expected column in data: {e}")
        raise
    except Exception as e:
        print("Error: Unexpected error occurred while processing the data.")
        print(e)
        raise


def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        # Create directories if they don't exist
        os.makedirs(data_path, exist_ok=True)

        # Save train and test data to CSV files
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        
        print(f"Data saved to {data_path}")
    except Exception as e:
        print(f"Error: Unexpected error occurred while saving the data to {data_path}.")
        print(e)
        raise


def main():
    try:
        # Load parameters for test size
        test_size = load_params(params_path="params.yaml")
        
        # Read the data from the URL
        df = read_data(url="https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv")

        # Process the data
        final_df = process_data(df)

        # Split the data into train and test sets (stratify to maintain class balance)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42, stratify=final_df['sentiment'])

        # Define the data path
        data_path = os.path.join("data", "raw")

        # Save the processed data
        save_data(data_path=data_path, train_data=train_data, test_data=test_data)
        
        print("Data ingestion process completed successfully.")

    except Exception as e:
        print(f"Error: {e}")
        print("Failed to complete the data ingestion process.")

if __name__ == "__main__":
    main()
