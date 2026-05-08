import argparse
import os
import wandb


def go(args):
    run = wandb.init(project="nyc_airbnb", job_type="download_data")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, "data", args.sample)

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )

    artifact.add_file(local_path)
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data and log it as a W&B artifact")
    parser.add_argument("sample", type=str, help="Name of the sample file to log")
    parser.add_argument("artifact_name", type=str, help="Name for the output artifact")
    parser.add_argument("artifact_type", type=str, help="Type for the output artifact")
    parser.add_argument("artifact_description", type=str, help="Description for the output artifact")

    args = parser.parse_args()
    go(args)
