import argparse
import logging
import json
import tempfile
import os
import sys

import mlflow
import pandas as pd
import wandb

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering import process_features


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S"
)

logger = logging.getLogger(__name__)


def go(args):
    run = wandb.init(project="nyc_airbnb", job_type="train_random_forest")
    run.config.update(vars(args))

    logger.info("Downloading and reading train artifact")
    artifact_path = run.use_artifact(args.trainval_artifact).file()
    df = pd.read_csv(artifact_path)

    logger.info("Splitting train/validation")
    X = df.copy()
    y = X.pop("price")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        random_state=args.random_seed,
        stratify=X[args.stratify_by]
    )

    logger.info("Loading random forest configuration")
    with open(args.rf_config, "r") as fp:
        rf_config = json.load(fp)

    categorical_features = ["neighbourhood_group", "room_type"]
    numerical_features = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]
    text_feature = "name"

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    text_transformer = Pipeline(
        steps=[
            ("selector", FunctionTransformer(process_features, validate=False)),
            ("tfidf", TfidfVectorizer(max_features=args.max_tfidf_features)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numerical_transformer, numerical_features),
            ("txt", text_transformer, text_feature),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("random_forest", RandomForestRegressor(**rf_config)),
        ]
    )

    logger.info("Training model")
    model.fit(X_train, y_train)

    logger.info("Inferring on validation set")
    preds = model.predict(X_val)

    rmse = mean_squared_error(y_val, preds, squared=False)
    logger.info(f"Validation RMSE: {rmse}")

    mlflow.log_metric("rmse", rmse)

    with tempfile.TemporaryDirectory() as tmp_dir:
        export_path = os.path.join(tmp_dir, "model_export")
        mlflow.sklearn.save_model(model, export_path)

        artifact = wandb.Artifact(
            args.output_artifact,
            type="model",
            description="Random forest model trained on trainval data",
        )
        artifact.add_dir(export_path)
        run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest model")

    parser.add_argument("--trainval_artifact", type=str, required=True)
    parser.add_argument("--rf_config", type=str, required=True)
    parser.add_argument("--val_size", type=float, required=True)
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--stratify_by", type=str, required=True)
    parser.add_argument("--max_tfidf_features", type=int, required=True)
    parser.add_argument("--output_artifact", type=str, required=True)

    args = parser.parse_args()
    go(args)
