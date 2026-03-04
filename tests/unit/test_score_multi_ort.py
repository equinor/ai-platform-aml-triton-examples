"""
Unit tests for src/triton_scoring/score_multi_ort.py.

All tests run without any AML / Azure infrastructure.
The AML stubs are installed by tests/conftest.py before this module is imported.
"""
import json
import os

import numpy as np
import pytest

import triton_scoring.score_multi_ort as mod
from tests.aml_stubs import AMLRequest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _req(body=b"", method="POST", path="/score", args=None):
    if isinstance(body, dict):
        body = json.dumps(body).encode()
    return AMLRequest(method=method, path=path, body=body, args=args)


def _parse(resp):
    return json.loads(resp.body)


def _iris_payload():
    return {
        "inputs": [{
            "name": "float_input",
            "shape": [1, 4],
            "datatype": "FP32",
            "data": [[5.1, 3.5, 1.4, 0.2]],
        }]
    }


def _sine_payload():
    return {
        "inputs": [{
            "name": "x",
            "shape": [1, 1],
            "datatype": "FP32",
            "data": [[1.5707963]],
        }]
    }


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def populated_registry(iris_model, sine_params, monkeypatch):
    """Inject both models into the module registry and disable stale-reload."""
    registry = {
        "iris_classifier": {
            "type": "sklearn",
            "model": iris_model,
            "path": "/fake/iris/model.pkl",
        },
        "pytorch_sine": {
            "type": "numpy_mlp",
            "path": "/fake/sine/model_params.npz",
            **sine_params,
        },
    }
    monkeypatch.setattr(mod, "_registry", registry)
    # float("inf") makes mtime <= _state_mtime always True → fast path in _reload_if_stale
    monkeypatch.setattr(mod, "_state_mtime", float("inf"))
    return mod


# ── _relu ─────────────────────────────────────────────────────────────────────

class TestRelu:
    def test_positive_values_pass_through(self):
        arr = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(mod._relu(arr), arr)

    def test_negative_values_become_zero(self):
        arr = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(mod._relu(arr), [0.0, 0.0, 2.0])


# ── _parse_v2_tensor (parametrized) ──────────────────────────────────────────

@pytest.mark.parametrize("inp", [
    {"name": "wrong_name", "shape": [1, 4], "data": [[1, 2, 3, 4]]},
    {"name": "float_input", "shape": [1, 4], "data": [[1, 2, 3]]},   # count mismatch
])
def test_parse_v2_tensor_rejects_invalid(inp):
    with pytest.raises(ValueError):
        mod._parse_v2_tensor(inp, "float_input", 4)


# ── Management endpoints ──────────────────────────────────────────────────────

class TestManagementEndpoints:
    def test_health_returns_200(self, populated_registry):
        resp = populated_registry.run(_req(path="/v2/health/ready"))
        assert resp.status_code == 200

    def test_health_status_is_ready(self, populated_registry):
        body = _parse(populated_registry.run(_req(path="/v2/health/ready")))
        assert body["status"] == "ready"

    def test_health_lists_both_models(self, populated_registry):
        body = _parse(populated_registry.run(_req(path="/v2/health/ready")))
        assert "iris_classifier" in body["models"]
        assert "pytorch_sine" in body["models"]

    def test_repo_index_lists_both_models(self, populated_registry):
        resp = populated_registry.run(_req(method="GET", path="/v2/repository/index"))
        names = [m["name"] for m in _parse(resp)["models"]]
        assert "iris_classifier" in names
        assert "pytorch_sine" in names

    def test_model_ready_loaded_returns_200(self, populated_registry):
        resp = populated_registry.run(_req(path="/v2/models/iris_classifier/ready"))
        assert resp.status_code == 200
        assert _parse(resp)["ready"] is True

    def test_model_ready_not_loaded_returns_404(self, populated_registry):
        resp = populated_registry.run(_req(path="/v2/models/no_such_model/ready"))
        assert resp.status_code == 404


# ── Inference routing ─────────────────────────────────────────────────────────

class TestInferenceRouting:
    def test_query_param_routes_iris(self, populated_registry):
        resp = populated_registry.run(
            _req(body=_iris_payload(), args={"model": "iris_classifier"})
        )
        assert resp.status_code == 200
        assert _parse(resp)["model_name"] == "iris_classifier"

    def test_query_param_routes_pytorch_sine(self, populated_registry):
        resp = populated_registry.run(
            _req(body=_sine_payload(), args={"model": "pytorch_sine"})
        )
        assert resp.status_code == 200
        assert _parse(resp)["model_name"] == "pytorch_sine"

    def test_path_routing(self, populated_registry):
        resp = populated_registry.run(
            _req(body=_iris_payload(), path="/v2/models/iris_classifier/infer")
        )
        assert resp.status_code == 200

    def test_body_field_routing(self, populated_registry):
        payload = {**_iris_payload(), "model_name": "iris_classifier"}
        resp = populated_registry.run(_req(body=payload))
        assert resp.status_code == 200

    def test_unknown_model_returns_404(self, populated_registry):
        resp = populated_registry.run(
            _req(body=_iris_payload(), args={"model": "no_such_model"})
        )
        assert resp.status_code == 404

    def test_no_routing_info_returns_400(self, populated_registry):
        resp = populated_registry.run(_req(body=_iris_payload()))
        assert resp.status_code == 400

    def test_get_on_inference_path_returns_405(self, populated_registry):
        resp = populated_registry.run(_req(method="GET", body=b""))
        assert resp.status_code == 405

    def test_bad_json_returns_400(self, populated_registry):
        resp = populated_registry.run(AMLRequest(body=b"not-valid-json{{"))
        assert resp.status_code == 400

    def test_invalid_tensor_returns_error(self, populated_registry):
        # shape [1, 4] but data has only 3 elements — _parse_v2_tensor raises ValueError.
        # _dispatch catches all exceptions with a broad except-block and returns 500.
        payload = {
            "inputs": [{"name": "float_input", "shape": [1, 4], "data": [[1, 2, 3]]}]
        }
        resp = populated_registry.run(
            _req(body=payload, args={"model": "iris_classifier"})
        )
        assert resp.status_code in {400, 500}


# ── Model control (load / unload) ─────────────────────────────────────────────

class TestModelControl:
    def test_unload_removes_model(self, populated_registry, monkeypatch, tmp_path):
        monkeypatch.setattr(mod, "_STATE_FILE", str(tmp_path / "state.json"))
        resp = populated_registry.run(
            _req(method="POST", path="/v2/repository/models/iris_classifier/unload")
        )
        assert resp.status_code == 200
        assert "iris_classifier" not in mod._registry

    def test_unload_unknown_returns_404(self, populated_registry, monkeypatch, tmp_path):
        monkeypatch.setattr(mod, "_STATE_FILE", str(tmp_path / "state.json"))
        resp = populated_registry.run(
            _req(method="POST", path="/v2/repository/models/no_such_model/unload")
        )
        assert resp.status_code == 404

    def test_load_restores_model(self, populated_registry, model_dir, monkeypatch, tmp_path):
        monkeypatch.setattr(mod, "_STATE_FILE", str(tmp_path / "state.json"))
        monkeypatch.setattr(mod, "_model_dir", model_dir)
        del mod._registry["iris_classifier"]
        resp = populated_registry.run(
            _req(method="POST", path="/v2/repository/models/iris_classifier/load")
        )
        assert resp.status_code == 200
        assert "iris_classifier" in mod._registry

    def test_load_unknown_model_returns_404(self, populated_registry, monkeypatch, tmp_path):
        monkeypatch.setattr(mod, "_STATE_FILE", str(tmp_path / "state.json"))
        resp = populated_registry.run(
            _req(method="POST", path="/v2/repository/models/no_such_model/load")
        )
        assert resp.status_code == 404


# ── _reload_if_stale ──────────────────────────────────────────────────────────

class TestReloadIfStale:
    def test_no_state_file_is_noop(self, populated_registry, monkeypatch, tmp_path):
        monkeypatch.setattr(mod, "_STATE_FILE", str(tmp_path / "does_not_exist.json"))
        monkeypatch.setattr(mod, "_state_mtime", 0.0)
        mod._reload_if_stale()
        # Registry should be unchanged
        assert "iris_classifier" in mod._registry

    def test_syncs_from_written_state_file(self, populated_registry, monkeypatch, tmp_path):
        state_file = str(tmp_path / "state.json")
        monkeypatch.setattr(mod, "_STATE_FILE", state_file)
        monkeypatch.setattr(mod, "_state_mtime", 0.0)
        monkeypatch.setattr(mod, "_model_dir", "")  # won't be needed (only unloading)

        # Write a state that only keeps pytorch_sine
        with open(state_file, "w") as f:
            json.dump({"loaded": ["pytorch_sine"]}, f)

        mod._reload_if_stale()
        assert "iris_classifier" not in mod._registry
        assert "pytorch_sine" in mod._registry


# ── init() ────────────────────────────────────────────────────────────────────

class TestInit:
    def test_both_models_loaded(self, model_dir, monkeypatch, tmp_path):
        monkeypatch.setenv("AZUREML_MODEL_DIR", model_dir)
        monkeypatch.setattr(mod, "_registry", {})
        monkeypatch.setattr(mod, "_STATE_FILE", str(tmp_path / "state.json"))
        mod.init()
        assert "iris_classifier" in mod._registry
        assert "pytorch_sine" in mod._registry

    def test_state_file_written(self, model_dir, monkeypatch, tmp_path):
        monkeypatch.setenv("AZUREML_MODEL_DIR", model_dir)
        monkeypatch.setattr(mod, "_registry", {})
        state_file = str(tmp_path / "registry.json")
        monkeypatch.setattr(mod, "_STATE_FILE", state_file)
        mod.init()
        assert os.path.exists(state_file)
