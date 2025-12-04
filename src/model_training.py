import os
import sys
import yaml 
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import create_logger

logger = create_logger('model_training')

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str, target_column: str = 'category') -> tuple[np.ndarray, np.ndarray]:
    """Load CSV data and separate features and target."""
    try:
        df = pd.read_csv(data_url, encoding='latin1')
        logger.debug("Data successfully loaded from %s", data_url)
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
        return X, y
    except Exception as e:
        logger.error("Failed to load data from %s: %s", data_url, e, exc_info=True)
        raise


def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict = None) -> SVC:
    """Train an SVC model with given hyperparameters."""
    try:
        params = params or {'kernel': 'linear', 'C': 1.0, 'random_state': 42}
        logger.debug("Training SVC model with parameters: %s", params)
        model = SVC(**params)
        model.fit(X_train, y_train)
        logger.debug("Model training completed.")
        return model
    except Exception as e:
        logger.error("Error during model training: %s", e, exc_info=True)
        raise


def save_model(model: SVC, file_path: str) -> None:
    """Save trained model to a file using pickle."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug("Model saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving model: %s", e, exc_info=True)
        raise


def main():
    try:
        train_file = './data/features/train_tfidf.csv'
        model_file = './models/svc_model.pkl'
        params = load_params('params.yaml')['model_training']
        X_train, y_train = load_data(train_file)

        model = train_model(X_train, y_train, params=params)


        save_model(model, model_file)

        logger.debug("Training pipeline completed successfully.")
    except Exception as e:
        logger.error("Error in training pipeline: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
