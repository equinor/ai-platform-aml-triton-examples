"""
Unit tests for src/iris_pipeline/train.py.

The script is exercised via subprocess so argparse and the real entrypoint
are tested end-to-end without mocking.
"""
import subprocess
import sys
from pathlib import Path

import joblib
import pytest


TRAIN_SCRIPT = (
    Path(__file__).parents[2] / "src" / "iris_pipeline" / "train.py"
)


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    """Run train.py once; return the loaded model and the output directory."""
    out = tmp_path_factory.mktemp("train_output")
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--output_dir", str(out)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"train.py failed:\n{result.stderr}"
    model = joblib.load(out / "model.pkl")
    return model, out


def test_model_pkl_created(trained_model):
    _, out = trained_model
    assert (out / "model.pkl").exists()


def test_model_is_random_forest(trained_model):
    from sklearn.ensemble import RandomForestClassifier
    model, _ = trained_model
    assert isinstance(model, RandomForestClassifier)


def test_accuracy_at_least_90_percent(trained_model):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    model, _ = trained_model
    iris = load_iris()
    _, X_test, _, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    assert model.score(X_test, y_test) >= 0.90


def test_n_features_is_4(trained_model):
    model, _ = trained_model
    assert model.n_features_in_ == 4


def test_three_classes(trained_model):
    model, _ = trained_model
    assert len(model.classes_) == 3
