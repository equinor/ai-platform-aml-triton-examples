"""
Unit tests for src/triton_scoring/score_ort.py.

All tests run without any AML / Azure infrastructure.
The AML stubs are installed by tests/conftest.py before this module is imported.
"""
import json

import numpy as np
import pytest

import triton_scoring.score_ort as mod
from tests.aml_stubs import AMLRequest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _req(body, method="POST"):
    if isinstance(body, dict):
        body = json.dumps(body).encode()
    return AMLRequest(method=method, body=body)


def _parse(resp):
    return json.loads(resp.body)


def _iris_payload(data=None, shape=None, name="float_input"):
    rows = data if data is not None else [[5.1, 3.5, 1.4, 0.2]]
    return {
        "inputs": [{
            "name": name,
            "shape": shape or [len(rows), 4],
            "datatype": "FP32",
            "data": rows,
        }]
    }


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def loaded(iris_model, monkeypatch):
    """Patch the module-level _model with the session iris_model."""
    monkeypatch.setattr(mod, "_model", iris_model)
    return mod


# ── _parse_v2_tensor ──────────────────────────────────────────────────────────

class TestParseV2Tensor:
    def test_valid_returns_float32_array(self):
        inp = {"name": "float_input", "shape": [1, 4], "data": [[5.1, 3.5, 1.4, 0.2]]}
        arr = mod._parse_v2_tensor(inp, "float_input", 4)
        assert arr.shape == (1, 4)
        assert arr.dtype == np.float32

    def test_flat_data_accepted(self):
        inp = {"name": "float_input", "shape": [1, 4], "data": [5.1, 3.5, 1.4, 0.2]}
        arr = mod._parse_v2_tensor(inp, "float_input", 4)
        assert arr.shape == (1, 4)

    def test_wrong_name_raises(self):
        inp = {"name": "bad_name", "shape": [1, 4], "data": [[1, 2, 3, 4]]}
        with pytest.raises(ValueError, match="Expected tensor 'float_input'"):
            mod._parse_v2_tensor(inp, "float_input", 4)

    def test_shape_data_mismatch_raises(self):
        # shape says 4 elements per row, data only has 3
        inp = {"name": "float_input", "shape": [1, 4], "data": [[1, 2, 3]]}
        with pytest.raises(ValueError, match="elements"):
            mod._parse_v2_tensor(inp, "float_input", 4)

    def test_batch_exceeds_max_raises(self):
        rows = [[1.0, 2.0, 3.0, 4.0]] * 1025
        inp = {"name": "float_input", "shape": [1025, 4], "data": rows}
        with pytest.raises(ValueError, match="Batch size"):
            mod._parse_v2_tensor(inp, "float_input", 4)

    def test_wrong_features_raises(self):
        inp = {"name": "float_input", "shape": [1, 3], "data": [[1, 2, 3]]}
        with pytest.raises(ValueError, match="features"):
            mod._parse_v2_tensor(inp, "float_input", 4)

    def test_nan_raises(self):
        inp = {"name": "float_input", "shape": [1, 4], "data": [[float("nan"), 1, 2, 3]]}
        with pytest.raises(ValueError, match="NaN or Inf"):
            mod._parse_v2_tensor(inp, "float_input", 4)

    def test_inf_raises(self):
        inp = {"name": "float_input", "shape": [1, 4], "data": [[float("inf"), 1, 2, 3]]}
        with pytest.raises(ValueError, match="NaN or Inf"):
            mod._parse_v2_tensor(inp, "float_input", 4)

    def test_missing_shape_raises(self):
        inp = {"name": "float_input", "data": [[1, 2, 3, 4]]}
        with pytest.raises(ValueError, match="missing 'shape' or 'data'"):
            mod._parse_v2_tensor(inp, "float_input", 4)


# ── init() ────────────────────────────────────────────────────────────────────

class TestInit:
    def test_loads_model_from_model_dir(self, model_dir, monkeypatch):
        monkeypatch.setenv("AZUREML_MODEL_DIR", model_dir)
        monkeypatch.setattr(mod, "_model", None)
        mod.init()
        assert mod._model is not None

    def test_raises_when_pkl_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AZUREML_MODEL_DIR", str(tmp_path))
        monkeypatch.setattr(mod, "_model", None)
        with pytest.raises(FileNotFoundError):
            mod.init()


# ── run() ─────────────────────────────────────────────────────────────────────

class TestRun:
    def test_returns_200(self, loaded):
        resp = loaded.run(_req(_iris_payload()))
        assert resp.status_code == 200

    def test_model_name_in_response(self, loaded):
        body = _parse(loaded.run(_req(_iris_payload())))
        assert body["model_name"] == "iris_classifier"

    def test_label_and_probabilities_in_outputs(self, loaded):
        body = _parse(loaded.run(_req(_iris_payload())))
        names = [o["name"] for o in body["outputs"]]
        assert "label" in names
        assert "probabilities" in names

    def test_label_in_valid_classes(self, loaded):
        body = _parse(loaded.run(_req(_iris_payload())))
        label_out = next(o for o in body["outputs"] if o["name"] == "label")
        assert label_out["data"][0] in {0, 1, 2}

    def test_probabilities_sum_to_one(self, loaded):
        body = _parse(loaded.run(_req(_iris_payload())))
        proba_out = next(o for o in body["outputs"] if o["name"] == "probabilities")
        assert abs(sum(proba_out["data"]) - 1.0) < 1e-5

    def test_batch_three_rows(self, loaded):
        rows = [[5.1, 3.5, 1.4, 0.2], [6.0, 2.9, 4.5, 1.5], [4.9, 3.1, 1.5, 0.1]]
        body = _parse(loaded.run(_req(_iris_payload(data=rows))))
        label_out = next(o for o in body["outputs"] if o["name"] == "label")
        assert len(label_out["data"]) == 3

    def test_get_returns_405(self, loaded):
        resp = loaded.run(_req(b"", method="GET"))
        assert resp.status_code == 405

    def test_bad_json_returns_400(self, loaded):
        resp = loaded.run(AMLRequest(body=b"not-valid-json{{"))
        assert resp.status_code == 400

    def test_missing_inputs_key_returns_400(self, loaded):
        resp = loaded.run(_req({"inputs": []}))
        assert resp.status_code == 400

    def test_wrong_tensor_name_returns_400(self, loaded):
        payload = _iris_payload(name="bad_tensor")
        resp = loaded.run(_req(payload))
        assert resp.status_code == 400

    def test_shape_data_mismatch_returns_400(self, loaded):
        # shape [1, 4] but only 3 elements — caught by _parse_v2_tensor
        payload = _iris_payload(data=[[1, 2, 3]], shape=[1, 4])
        resp = loaded.run(_req(payload))
        assert resp.status_code == 400

    def test_oversized_batch_returns_400(self, loaded):
        rows = [[1.0, 2.0, 3.0, 4.0]] * 1025
        payload = _iris_payload(data=rows, shape=[1025, 4])
        resp = loaded.run(_req(payload))
        assert resp.status_code == 400

    def test_model_exception_returns_500(self, loaded, monkeypatch):
        def _explode(X):
            raise RuntimeError("model exploded")

        monkeypatch.setattr(mod._model, "predict", _explode)
        resp = loaded.run(_req(_iris_payload()))
        assert resp.status_code == 500
