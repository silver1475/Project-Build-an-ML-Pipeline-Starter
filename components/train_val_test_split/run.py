#!/usr/bin/env python
"""
This script splits the provided dataframe into trainval and test
"""
import argparse
import logging
import os
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="train_val_test_split")
    run.config.update(vars(args))

    logger.info(f"Fetching artifact {args.input}")

    project_root = os.getcwd()
    local_input = os.path.join(project_root, "raw_data.csv")

    logger.info(
        f"WANDB_MODE={os.environ.get('WANDB_MODE')}, "
        f"local_input={local_input}, exists={os.path.exists(local_input)}"
    )

    if os.environ.get("WANDB_MODE", "").lower() == "disabled" and os.path.exists(local_input):
        artifact_local_path = local_input
    else:
        artifact = run.use_artifact(args.input)
        if artifact is None:
            raise FileNotFoundError(
                f"Could not fetch artifact {args.input} and local fallback file was not found at {local_input}"
            )
        artifact_local_path = artifact.file()

    df = pd.read_csv(artifact_local_path)

    logger.info("Splitting trainval and test")
    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != "none" else None,
    )

    trainval_path = os.path.join(project_root, "trainval_data.csv")
    test_path = os.path.join(project_root, "test_data.csv")

    trainval.to_csv(trainval_path, index=False)
    test.to_csv(test_path, index=False)

    logger.info(f"Saved trainval split to {trainval_path}")
    logger.info(f"Saved test split to {test_path}")

    logger.info("Uploading trainval_data.csv dataset")
    log_artifact(
        "trainval_data.csv",
        "trainval_data",
        "train/validation split of dataset",
        trainval_path,
        run,
    )

    logger.info("Uploading test_data.csv dataset")
    log_artifact(
        "test_data.csv",
        "test_data",
        "test split of dataset",
        test_path,
        run,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument("input", type=str, help="Input artifact to split")

    parser.add_argument(
        "test_size",
        type=float,
        help="Size of the test split. Fraction of the dataset, or number of items",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none",
        required=False,
    )

    args = parser.parse_args()
    go(args)
