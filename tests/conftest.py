"""
Pytest configuration for ai-platform-aml-triton-examples test suite.

Stubs for the AML inference server packages are injected into sys.modules at
module-import time (before any test file is collected) so that the scoring
scripts can import AMLRequest / AMLResponse / rawhttp without the real
azureml packages being installed.
"""
import sys
import types

import numpy as np
import pytest


# ── Install AML stubs ──────────────────────────────────────────────────────────
# Must run at module level so stubs are present before scoring scripts are
# imported during test collection.

def _install_aml_stubs() -> None:
    from tests.aml_stubs import AMLRequest, AMLResponse, rawhttp  # noqa: F401

    req_mod = types.ModuleType("azureml_inference_server_http.api.aml_request")
    req_mod.AMLRequest = AMLRequest
    req_mod.rawhttp = rawhttp

    resp_mod = types.ModuleType("azureml_inference_server_http.api.aml_response")
    resp_mod.AMLResponse = AMLResponse

    if "azureml_inference_server_http" not in sys.modules:
        sys.modules["azureml_inference_server_http"] = types.ModuleType(
            "azureml_inference_server_http"
        )
    if "azureml_inference_server_http.api" not in sys.modules:
        sys.modules["azureml_inference_server_http.api"] = types.ModuleType(
            "azureml_inference_server_http.api"
        )
    sys.modules["azureml_inference_server_http.api.aml_request"] = req_mod
    sys.modules["azureml_inference_server_http.api.aml_response"] = resp_mod


_install_aml_stubs()


# ── Session-level fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def iris_model():
    """Tiny 3-estimator RandomForestClassifier trained on the Iris dataset."""
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier

    iris = load_iris()
    clf = RandomForestClassifier(n_estimators=3, random_state=0)
    clf.fit(iris.data, iris.target)
    return clf


@pytest.fixture(scope="session")
def sine_params():
    """Random MLP weights compatible with the pytorch_sine numpy forward pass.

    Architecture: input(1) → hidden(8) → hidden(8) → output(1)
    Shapes mirror what _load_pytorch_sine expects in model_params.npz.
    """
    rng = np.random.default_rng(42)
    hidden = 8
    return {
        "W1": rng.standard_normal((hidden, 1)).astype(np.float32),
        "b1": rng.standard_normal(hidden).astype(np.float32),
        "W2": rng.standard_normal((hidden, hidden)).astype(np.float32),
        "b2": rng.standard_normal(hidden).astype(np.float32),
        "W3": rng.standard_normal((1, hidden)).astype(np.float32),
        "b3": rng.standard_normal(1).astype(np.float32),
    }


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, iris_model, sine_params):
    """Temporary directory with the Triton model-repo layout.

    iris_classifier/1/model.pkl
    pytorch_sine/1/model_params.npz
    """
    import joblib

    base = tmp_path_factory.mktemp("models")

    iris_dir = base / "iris_classifier" / "1"
    iris_dir.mkdir(parents=True)
    joblib.dump(iris_model, iris_dir / "model.pkl")

    sine_dir = base / "pytorch_sine" / "1"
    sine_dir.mkdir(parents=True)
    np.savez(sine_dir / "model_params.npz", **sine_params)

    return str(base)
