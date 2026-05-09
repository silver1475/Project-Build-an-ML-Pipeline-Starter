import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S"
)


def go(args):

    run = wandb.init(project="nyc_airbnb", job_type="basic_cleaning")

    run.config.update(args)

    logging.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logging.info("Reading dataframe")
    df = pd.read_csv(artifact_local_path)

    logging.info("Removing outliers")
    df = df[df.price.between(args.min_price, args.max_price)]

    logging.info("Converting last_review to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"])

    logging.info("Removing outliers in longitude and latitude")
    df = df[
        df["longitude"].between(-74.25, -73.50) &
        df["latitude"].between(40.5, 40.95)
    ].copy()

    logging.info("Saving cleaned data")
    df.to_csv("clean_sample.csv", index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    artifact.add_file("clean_sample.csv")

    logging.info("Logging artifact")
    run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        required=True,
        help="Input artifact to clean"
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        required=True,
        help="Name for the output artifact"
    )

    parser.add_argument(
        "--output_type",
        type=str,
        required=True,
        help="Type for the output artifact"
    )

    parser.add_argument(
        "--output_description",
        type=str,
        required=True,
        help="Description for the output artifact"
    )

    parser.add_argument(
        "--min_price",
        type=float,
        required=True,
        help="Minimum price"
    )

    parser.add_argument(
        "--max_price",
        type=float,
        required=True,
        help="Maximum price"
    )

    args = parser.parse_args()

    go(args)