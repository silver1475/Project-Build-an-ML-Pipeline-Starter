import os
import pytest
import pandas as pd
import wandb


def pytest_addoption(parser):
    parser.addoption(
        "--csv",
        action="store",
        default="trainval_data.csv",
        help="Path to local CSV file or W&B artifact name",
    )


@pytest.fixture(scope="session")
def data(request):
    csv_arg = request.config.getoption("--csv")

    if os.path.exists(csv_arg):
        return pd.read_csv(csv_arg)

    run = wandb.init(job_type="data_checks")
    artifact = run.use_artifact(csv_arg)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path)
    run.finish()

    return df


@pytest.fixture(scope="session")
def ref_data():
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
    ref_path = os.path.join(project_root, "test_data.csv")
    return pd.read_csv(ref_path)


@pytest.fixture(scope="session")
def kl_threshold():
    return 0.2
