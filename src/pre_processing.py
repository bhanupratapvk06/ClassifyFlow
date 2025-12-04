import os
import sys
import spacy
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import create_logger

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

logger = create_logger('data_processing')


def transform_text(text: str) -> str:
    """Transform text: lowercase, remove stopwords/punct/space, lemmatize"""
    try:
        text = str(text).lower()
        doc = nlp(text)
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]
        return " ".join(tokens)
    except Exception as e:
        logger.error("Error in transforming text: %s", e, exc_info=True)
        raise


def preprocess_df(df: pd.DataFrame, text_column='text', target_column='label') -> pd.DataFrame:
    """Preprocess a DataFrame: encode labels and clean text."""
    try:
        logger.debug("Starting preprocessing for DataFrame...")

        if target_column in df.columns:
            le = LabelEncoder()
            df[target_column] = le.fit_transform(df[target_column])
            logger.debug("Target column encoded")
        else:
            logger.warning("Target column not found; skipping encoding")

        if text_column in df.columns:
            df[text_column] = df[text_column].apply(transform_text)
            logger.debug("Text column transformed")
        else:
            logger.error("Text column not found in DataFrame")
            raise ValueError(f"Column '{text_column}' does not exist in the DataFrame")

        return df

    except Exception as e:
        logger.error("Error during preprocessing DataFrame: %s", e, exc_info=True)
        raise


def main():
    """Main function to load raw data, preprocess it, and save processed data."""
    try:
        raw_path = './data/raw'
        processed_path = './data/processed'
        text_column = 'resume_str'
        target_column = 'category'

        os.makedirs(processed_path, exist_ok=True)

        train_data = pd.read_csv(os.path.join(raw_path, 'train.csv'))
        test_data = pd.read_csv(os.path.join(raw_path, 'test.csv'))
        logger.debug("Raw data loaded successfully")

        train_processed = preprocess_df(train_data, text_column, target_column)
        test_processed = preprocess_df(test_data, text_column, target_column)

        train_processed.to_csv(os.path.join(processed_path, 'train.csv'), index=False)
        test_processed.to_csv(os.path.join(processed_path, 'test.csv'), index=False)
        logger.debug(f"Processed data saved to {processed_path}")

    except Exception as e:
        logger.error("Error in main pipeline: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
