import numpy as np
import pandas as pd
import os
import re
import string
import nltk
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# logging configure
logger = logging.getLogger('data_preprocessing')
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

nltk.download('wordnet')
nltk.download('stopwords')

def load_data(train_data_path: str, test_data_path: str) -> tuple:
    try:
        logger.info(f"Loading training data from {train_data_path}")
        train_data = pd.read_csv(train_data_path)
        logger.info(f"Loading test data from {test_data_path}")
        test_data = pd.read_csv(test_data_path)
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def lemmatization(text: str) -> str:
    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(y) for y in text]
        return " ".join(text)
    except Exception as e:
        logger.error(f"Error during lemmatization: {e}")
        return text

def remove_stop_words(text: str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
        text = [i for i in str(text).split() if i not in stop_words]
        return " ".join(text)
    except Exception as e:
        logger.error(f"Error removing stop words: {e}")
        return text

def removing_numbers(text: str) -> str:
    try:
        return ''.join([i for i in text if not i.isdigit()])
    except Exception as e:
        logger.error(f"Error removing numbers: {e}")
        return text

def lower_case(text: str) -> str:
    try:
        return text.lower()
    except Exception as e:
        logger.error(f"Error converting to lowercase: {e}")
        return text

def removing_punctuations(text: str) -> str:
    try:
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        text = re.sub('\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.error(f"Error removing punctuations: {e}")
        return text

def removing_urls(text: str) -> str:
    try:
        return re.sub(r'https?://\S+|www\.\S+', '', text)
    except Exception as e:
        logger.error(f"Error removing URLs: {e}")
        return text

def remove_small_sentences(df: pd.DataFrame) -> None:
    try:
        df.loc[df['text'].str.split().str.len() < 3, 'text'] = np.nan
        logger.info("Removed small sentences with fewer than 3 words.")
    except Exception as e:
        logger.error(f"Error removing small sentences: {e}")

def remove_null_values(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.dropna(inplace=True)
        logger.info("Removed null values from dataset.")
        return df
    except Exception as e:
        logger.error(f"Error removing null values: {e}")
        return df

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        
        processed_df = remove_null_values(df)
        logger.info("Text normalization completed successfully.")
        return processed_df
    except Exception as e:
        logger.error(f"Error normalizing text: {e}")
        return df

def save_processed_data(data_path: str, train_processed_data: pd.DataFrame, test_processed_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        logger.info("Processed data saved successfully.")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")

def main():
    try:
        train_data, test_data = load_data('./data/raw/train.csv', './data/raw/test.csv')
        
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)
        
        save_processed_data("data/interim", train_processed_data, test_processed_data)
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")

if __name__ == "__main__":
    main()