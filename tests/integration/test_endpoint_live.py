"""
Integration smoke tests against a live AML Kubernetes Online Endpoint.

All tests are skipped unless the following environment variables are set:
  ENDPOINT_URL  — the scoring URI  (e.g. https://<endpoint>.<region>.inference.ml.azure.com/score)
  API_KEY       — the endpoint primary or secondary key

Optional:
  VERIFY_SSL    — set to "1" to enable SSL certificate verification (default: off).
                  The aurora.equinor.com gateway uses a private CA cert that is not
                  in the default system trust store, so verification is disabled by
                  default (equivalent to curl -k / requests verify=False).

Run with:
    pytest tests/integration/ -m integration -v
"""
import json
import os
import ssl
import urllib.error
import urllib.request

import pytest

ENDPOINT_URL = os.environ.get("ENDPOINT_URL", "")
API_KEY = os.environ.get("API_KEY", "")
_VERIFY_SSL = os.environ.get("VERIFY_SSL", "0") == "1"

_skip = pytest.mark.skipif(
    not ENDPOINT_URL or not API_KEY,
    reason="ENDPOINT_URL and API_KEY environment variables must be set",
)

# SSL context — skip verification unless VERIFY_SSL=1 is set.
# The AKS gateway (unified.aurora.equinor.com) terminates TLS with a corporate
# CA certificate that is not present in the default Python trust store.
_ssl_ctx = ssl.create_default_context()
if not _VERIFY_SSL:
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = ssl.CERT_NONE

_https_handler = urllib.request.HTTPSHandler(context=_ssl_ctx)
_opener = urllib.request.build_opener(_https_handler)


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _post(path="", body=None, key=API_KEY, url_override=None):
    """Send a POST request to the endpoint; returns (status_code, body_dict)."""
    target = url_override or (ENDPOINT_URL.rstrip("/") + ("/" + path.lstrip("/") if path else ""))
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        target,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with _opener.open(req) as r:
            raw = r.read().decode()
            # AML double-encodes the run() return value: response.text is a
            # JSON-encoded string whose value is itself a JSON object string.
            try:
                parsed = json.loads(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                parsed = json.loads(raw)
            return r.status, parsed
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            body_dict = json.loads(raw)
        except json.JSONDecodeError:
            body_dict = {"raw": raw}
        return e.code, body_dict


# ── iris_classifier ──────────────────────────────────────────────────────────

@pytest.mark.integration
class TestIrisClassifier:
    """Tests for the iris_classifier model.

    Uses body-field routing (model_name key) so these tests work against both
    the single-model endpoint (score_ort.py — model_name field is ignored) and
    the multi-model endpoint (score_multi_ort.py — model_name field is required).
    """

    @_skip
    def test_single_row_returns_200(self):
        payload = {
            "model_name": "iris_classifier",
            "inputs": [{
                "name": "float_input",
                "shape": [1, 4],
                "datatype": "FP32",
                "data": [[5.1, 3.5, 1.4, 0.2]],
            }],
        }
        status, body = _post(body=payload)
        assert status == 200

    @_skip
    def test_predicted_class_is_valid(self):
        payload = {
            "model_name": "iris_classifier",
            "inputs": [{
                "name": "float_input",
                "shape": [1, 4],
                "datatype": "FP32",
                "data": [[5.1, 3.5, 1.4, 0.2]],
            }],
        }
        _, body = _post(body=payload)
        label_out = next(o for o in body["outputs"] if o["name"] == "label")
        assert label_out["data"][0] in {0, 1, 2}

    @_skip
    def test_probabilities_sum_to_one(self):
        payload = {
            "model_name": "iris_classifier",
            "inputs": [{
                "name": "float_input",
                "shape": [1, 4],
                "datatype": "FP32",
                "data": [[6.3, 3.3, 4.7, 1.6]],
            }],
        }
        _, body = _post(body=payload)
        proba_out = next(o for o in body["outputs"] if o["name"] == "probabilities")
        assert abs(sum(proba_out["data"]) - 1.0) < 1e-4

    @_skip
    def test_batch_of_three_rows(self):
        rows = [[5.1, 3.5, 1.4, 0.2], [6.0, 2.9, 4.5, 1.5], [6.3, 3.3, 6.0, 2.5]]
        payload = {
            "model_name": "iris_classifier",
            "inputs": [{
                "name": "float_input",
                "shape": [3, 4],
                "datatype": "FP32",
                "data": rows,
            }],
        }
        status, body = _post(body=payload)
        assert status == 200
        label_out = next(o for o in body["outputs"] if o["name"] == "label")
        assert len(label_out["data"]) == 3


# ── pytorch_sine ─────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestPytorchSine:
    @_skip
    def test_sine_of_half_pi_approx_one(self):
        """sin(π/2) ≈ 1.0 — tests a well-trained model."""
        import math
        payload = {
            "model_name": "pytorch_sine",
            "inputs": [{"name": "x", "shape": [1, 1], "datatype": "FP32",
                        "data": [[math.pi / 2]]}],
        }
        status, body = _post(body=payload)
        assert status == 200
        y_out = next(o for o in body["outputs"] if o["name"] == "y")
        assert abs(y_out["data"][0] - 1.0) < 0.1


# ── Error handling ────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestErrorHandling:
    @_skip
    def test_unknown_model_returns_4xx(self):
        payload = {
            "model_name": "does_not_exist",
            "inputs": [{"name": "float_input", "shape": [1, 4], "data": [[1, 2, 3, 4]]}],
        }
        status, _ = _post(body=payload)
        assert status in {400, 404}

    @_skip
    def test_shape_mismatch_returns_error(self):
        # shape says 4 features, data only has 3 — invalid tensor.
        # score_multi_ort._dispatch catches ValueError with a broad except-block
        # and returns 500; the AKS gateway may surface this as 502 Bad Gateway.
        payload = {
            "model_name": "iris_classifier",
            "inputs": [{"name": "float_input", "shape": [1, 4], "data": [[1, 2, 3]]}],
        }
        status, _ = _post(body=payload)
        assert status in {400, 500, 502}

    @_skip
    def test_missing_auth_returns_401_or_403(self):
        payload = {
            "model_name": "iris_classifier",
            "inputs": [{"name": "float_input", "shape": [1, 4], "data": [[1, 2, 3, 4]]}],
        }
        status, _ = _post(body=payload, key="wrong-key")
        assert status in {401, 403}
