import os
import subprocess
import hydra


@hydra.main(config_path=".", config_name="config", version_base=None)
def go(config):

    root_path = hydra.utils.get_original_cwd()
    python = os.path.join(root_path, "venv", "bin", "python")

    steps = config["main"]["steps"].split(",")

    if "download" in steps:
        subprocess.run(
            [
                python,
                os.path.join(root_path, "components", "get_data", "run.py"),
                "sample1.csv",
                "raw_data.csv",
                "raw_data",
                "Raw file as downloaded",
            ],
            check=True,
        )

    if "basic_cleaning" in steps:
        subprocess.run(
            [
                python,
                os.path.join(root_path, "src", "basic_cleaning", "run.py"),
                "--input_artifact",
                f"{config['main']['project_name']}/raw_data.csv:latest",
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
            ],
            check=True,
        )

    if "data_split" in steps:
        subprocess.run(
            [
                python,
                os.path.join(root_path, "components", "train_val_test_split", "run.py"),
                f"{config['main']['project_name']}/clean_sample.csv:latest",
                str(config["modeling"]["test_size"]),
                "--random_seed",
                str(config["modeling"]["random_seed"]),
                "--stratify_by",
                config["modeling"]["stratify_by"],
            ],
            check=True,
        )


if __name__ == "__main__":
    go()
