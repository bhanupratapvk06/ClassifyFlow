import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import create_logger

logger = create_logger('feature_engineering')


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from CSV file with proper logging."""
    try:
        df = pd.read_csv(data_url, encoding='latin1')
        logger.debug("Data successfully loaded from %s", data_url)
        return df

    except pd.errors.ParserError as e:
        logger.error("CSV parsing failed for %s: %s", data_url, e)
        raise

    except FileNotFoundError as e:
        logger.error("File not found: %s", data_url)
        raise

    except Exception as e:
        logger.error("Unexpected error while loading %s: %s", data_url, e)
        raise


def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, text_column: str, target_column: str, max_features: int = 5000) -> tuple:
    """Apply TF-IDF to train and test data."""
    try:

        for col in [text_column, target_column]:
            if col not in train_data.columns or col not in test_data.columns:
                raise ValueError(f"Column '{col}' not found in train/test data")


        train_data[text_column] = train_data[text_column].fillna('')
        test_data[text_column] = test_data[text_column].fillna('')

        tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))


        X_train_tfidf = tfidf.fit_transform(train_data[text_column])
        X_test_tfidf = tfidf.transform(test_data[text_column])

        y_train = train_data[target_column].values
        y_test = test_data[target_column].values

        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df[target_column] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df[target_column] = y_test

        logger.debug("TFIDF applied and data transformed")
        return train_df, test_df

    except Exception as e:
        logger.error("Error during TFIDF transformation: %s", e, exc_info=True)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug("Data saved to %s", file_path)

    except Exception as e:
        logger.error("Unexpected error occurred while saving data: %s", e, exc_info=True)
        raise


def main():
    try:
        text_column = 'resume_str'
        target_column = 'category'
        max_features = 5000

        train_data = load_data('./data/processed/train.csv')
        test_data = load_data('./data/processed/test.csv')

        train_tfidf, test_tfidf = apply_tfidf(train_data, test_data, text_column, target_column, max_features)

        save_data(train_tfidf, './data/features/train_tfidf.csv')
        save_data(test_tfidf, './data/features/test_tfidf.csv')

        logger.debug("Feature engineering pipeline completed successfully.")

    except Exception as e:
        logger.error("Error in feature engineering pipeline: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
