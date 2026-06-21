"""
Generic training script — this file you should NOT need to edit.
It calls into model.py for anything project-specific.

Run with: python src/train.py
"""

import yaml
import logging
import mlflow

from model import (
    build_model,
    load_data,
    train,
    evaluate,
    save_model,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    data_cfg     = config["data"]
    model_cfg    = config["model"]
    training_cfg = config["training"]
    mlflow_cfg   = config["mlflow"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run():
        logging.info("Loading data...")
        X_train, X_test, y_train, y_test = load_data(
            data_cfg["train_path"],
            data_cfg["test_size"],
            data_cfg["random_state"],
        )

        logging.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        mlflow.log_params(training_cfg["hyperparameters"])
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.set_tag("model_type", model_cfg["type"])

        logging.info("Building model...")
        model = build_model(training_cfg["hyperparameters"])

        logging.info("Training...")
        model = train(model, X_train, y_train)

        logging.info("Evaluating...")
        metrics = evaluate(model, X_test, y_test)

        for name, value in metrics.items():
            logging.info(f"  {name}: {value:.4f}")
        mlflow.log_metrics(metrics)

        logging.info(f"Saving model to {model_cfg['save_path']}")
        save_model(model, model_cfg["save_path"])
        mlflow.log_artifact(model_cfg["save_path"])

        logging.info("Done. Run `mlflow ui` to view results.")


if __name__ == "__main__":
    main()