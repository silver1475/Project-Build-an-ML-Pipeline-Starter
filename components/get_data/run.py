import argparse
import os
import requests
import wandb


def go(args):
    run = wandb.init(project="nyc_airbnb", job_type="download_data")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    if args.sample.startswith("http://") or args.sample.startswith("https://"):
        filename = os.path.basename(args.sample)
        local_path = os.path.join(data_dir, filename)

        r = requests.get(args.sample)
        r.raise_for_status()

        with open(local_path, "wb") as f:
            f.write(r.content)
    else:
        local_path = os.path.join(data_dir, args.sample)

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
    parser.add_argument("sample", type=str, help="URL or local sample file name")
    parser.add_argument("artifact_name", type=str, help="Name for the output artifact")
    parser.add_argument("artifact_type", type=str, help="Type for the output artifact")
    parser.add_argument("artifact_description", type=str, help="Description for the output artifact")

    args = parser.parse_args()
    go(args)
