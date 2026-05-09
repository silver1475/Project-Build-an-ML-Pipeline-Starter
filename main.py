import os
import subprocess
import mlflow
import hydra
from omegaconf import DictConfig


def _run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


@hydra.main(config_path=".", config_name="config", version_base=None)
def go(config: DictConfig):

    project_name = config["main"]["project_name"]
    experiment_name = config["main"]["experiment_name"]
    active_steps = config["main"]["steps"].split(",")

    # Set the W&B project name from config
    os.environ["WANDB_PROJECT"] = project_name

    if experiment_name != "null":
        mlflow.set_experiment(experiment_name)

    # 1. Download Step - Now correctly references config["data"]["file_url"]
    if "download" in active_steps:
        _run(
            [
                "python",
                "components/get_data/run.py",
                config["data"]["file_url"],
                "raw_data.csv",
                "raw_data",
                "Raw file as downloaded",
            ]
        )

    # 2. Basic Cleaning Step
    if "basic_cleaning" in active_steps:
        _run(
            [
                "python",
                "src/basic_cleaning/run.py",
                "--input_artifact",
                "raw_data.csv:latest",
                "--output_artifact",
                "clean_sample.csv",
                "--output_type",
                "clean_sample",
                "--output_description",
                "Data with outliers and null values removed",
                "--min_price",
                str(config["etl"]["min_price"]),
                "--max_price",
                str(config["etl"]["max_price"]),
            ]
        )

    # 3. Data Split Step
    if "data_split" in active_steps:
        _run(
            [
                "python",
                "components/train_val_test_split/run.py",
                "clean_sample.csv:latest",
                str(config["modeling"]["test_size"]),
                "--random_seed",
                str(config["modeling"]["random_seed"]),
                "--stratify_by",
                config["modeling"]["stratify_by"],
            ]
        )

    # 4. Train Random Forest Step
    if "train_random_forest" in active_steps:
        _run(
            [
                "python",
                "src/train_random_forest/run.py",
                "--trainval_artifact",
                "trainval_data.csv:latest",
                "--val_size",
                str(config["modeling"]["val_size"]),
                "--random_seed",
                str(config["modeling"]["random_seed"]),
                "--stratify_by",
                config["modeling"]["stratify_by"],
                "--max_tfidf_features",
                str(config["modeling"]["max_tfidf_features"]),
                "--rf_config",
                "src/train_random_forest/rf_config.json",
                "--output_artifact",
                "random_forest_export",
            ]
        )


if __name__ == "__main__":
    go()