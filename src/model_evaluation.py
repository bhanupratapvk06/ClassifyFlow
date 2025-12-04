import os
import sys
import pickle
import json
import pandas as pd
import yaml
from dvclive import Live
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import create_logger

logger = create_logger('model_evaluation')

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


def load_model(file_path: str):
    """Load the Pickle model."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug("Model loaded from %s", file_path)
        return model
    except FileNotFoundError:
        logger.error("File not found at %s", file_path)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the model: %s", e, exc_info=True)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """Load CSV data."""
    try:
        df = pd.read_csv(data_url, encoding='latin1')
        logger.debug("Data successfully loaded from %s", data_url)
        return df
    except Exception as e:
        logger.error("Error loading data from %s: %s", data_url, e, exc_info=True)
        raise


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate the model and return metrics as a dictionary."""
    try:
        y_pred = model.predict(X_test)

        auc_score = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_prob)
        else:
            logger.warning("Model does not support predict_proba; skipping AUC computation")

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'auc': auc_score
        }
        logger.debug("Model evaluation metrics: %s", metrics)
        return metrics
    except Exception as e:
        logger.error("Error during model evaluation: %s", e, exc_info=True)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if not file_path.endswith('.json'):
            file_path = file_path.rsplit('.', 1)[0] + '.json'
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug("Metrics saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving metrics: %s", e, exc_info=True)
        raise


def main():
    try:
        model_file = './models/svc_model.pkl'
        test_file = './data/features/test_tfidf.csv'
        metrics_file = './reports/evaluation_metrics.json'
        params = load_params(params_path='params.yaml')
        model = load_model(model_file)
        test_df = load_data(test_file)
        X_test = test_df.drop('category', axis=1).values
        y_test = test_df['category'].values

        metrics = evaluate_model(model, X_test, y_test)
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test))
            live.log_metric('precision', precision_score(y_test, y_test))
            live.log_metric('recall', recall_score(y_test, y_test))

            live.log_params(params)
        save_metrics(metrics, metrics_file)

        logger.debug("Model evaluation pipeline completed successfully.")
    except Exception as e:
        logger.error("Error in evaluation pipeline: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
