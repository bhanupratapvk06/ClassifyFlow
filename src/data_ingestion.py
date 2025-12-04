import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import create_logger

logger = create_logger('data_ingestion')


def load_data(data_url: str, encoding: str = 'latin1') -> pd.DataFrame:
    """Load data from CSV file with proper logging."""
    try:
        df = pd.read_csv(data_url, encoding=encoding)
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


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)

        logger.debug("Train and Test data saved to %s", raw_data_path)

    except Exception as e:
        logger.error("Unexpected error occurred while saving data: %s", e)
        raise


def preprocess_data(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    """Generic preprocessing function for any dataset."""
    try:
        df = df[[text_col, label_col]].copy()
        df.drop_duplicates(inplace=True)
        df.dropna(subset=[text_col, label_col], inplace=True)
        df[text_col] = df[text_col].str.lower().str.strip()

        df.rename(columns={text_col: 'resume_str', label_col: 'category'}, inplace=True)

        logger.debug("Preprocessing completed. Columns now: %s", df.columns.tolist())
        return df

    except Exception as e:
        logger.error("Error occurred during preprocessing: %s", e, exc_info=True)
        raise


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split dataframe into train and test datasets."""
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['category'])
    return train_data, test_data


def main():
    try:
        test_size = 0.2
        random_state = 42
        data_url = './data/Resume.csv'
        text_col = 'Resume_str'
        label_col = 'Category'
        save_path = './data'

        df = load_data(data_url)
        df = preprocess_data(df, text_col=text_col, label_col=label_col)
        train_data, test_data = split_data(df, test_size=test_size, random_state=random_state)
        save_data(train_data, test_data, save_path)

        logger.debug("Data ingestion pipeline completed successfully.")

    except Exception as e:
        logger.error("Fatal error in data ingestion pipeline: %s", e)
        raise


if __name__ == "__main__":
    main()
