"""
KFServing V2 compatible scoring script for AML Kubernetes Online Endpoints.

Loads the sklearn RandomForestClassifier directly and exposes the same
KFServing V2 inference protocol that Triton Inference Server would use.

Why not Triton here?
  nvcr/nvidia/tritonserver images are compiled against CUDA and require
  NVIDIA GPU drivers on the host.  CPU-only AKS nodes have no NVIDIA drivers,
  so tritonserver crashes (CrashLoopBackOff) at startup.  This script uses
  scikit-learn + joblib instead — zero CUDA dependency, proven to run on the
  cpu-2 compute pool.

API compatibility:
  Request  → KFServing V2 JSON  {"inputs": [{"name": "float_input", ...}]}
  Response → KFServing V2 JSON  {"outputs": [{"name": "label", ...}, ...]}
  This is identical to what the Triton ONNX Runtime backend would return.
"""
import glob
import json
import logging
import os
import time
import uuid

import joblib
import numpy as np

# ── AML rawhttp decorator ─────────────────────────────────────────────────────
# The azureml-inference-server-http package is always present inside AML
# inference containers — it is the server itself.  Try the newer public API
# first; fall back to the legacy azureml.contrib path for older images.
try:
    from azureml_inference_server_http.api.aml_request import AMLRequest, rawhttp
    from azureml_inference_server_http.api.aml_response import AMLResponse
    _RAWHTTP_SOURCE = "azureml_inference_server_http"
except ImportError:
    try:
        from azureml.contrib.services.aml_request import AMLRequest, rawhttp
        from azureml.contrib.services.aml_response import AMLResponse
        _RAWHTTP_SOURCE = "azureml.contrib.services"
    except ImportError as exc:
        raise ImportError(
            "Could not import rawhttp / AMLResponse from either "
            "'azureml_inference_server_http' or 'azureml.contrib.services'. "
            "Ensure the AML inference server package is installed."
        ) from exc

logger = logging.getLogger(__name__)

MODEL_NAME  = "iris_classifier"
_MAX_BATCH  = 1024   # guard against unreasonably large request payloads
_N_FEATURES = 4      # iris: sepal length/width, petal length/width

_model = None


# ── Init ──────────────────────────────────────────────────────────────────────

def init():
    """Called once at container start by the AML inference server."""
    global _model

    model_dir = os.environ.get("AZUREML_MODEL_DIR", "")
    logger.info(f"[init] AZUREML_MODEL_DIR = {model_dir!r}")
    logger.info(f"[init] rawhttp source     = {_RAWHTTP_SOURCE!r}")

    # Locate model.pkl inside the Triton model repository structure.
    # Expected layout:  {AZUREML_MODEL_DIR}/iris_classifier/1/model.pkl
    pkl_candidates = glob.glob(
        os.path.join(model_dir, "**", "model.pkl"), recursive=True
    )
    if not pkl_candidates:
        raise FileNotFoundError(
            f"model.pkl not found under {model_dir!r}. "
            f"Contents: {os.listdir(model_dir) if os.path.isdir(model_dir) else 'N/A'}"
        )

    pkl_path = pkl_candidates[0]
    logger.info(f"[init] Loading sklearn model from: {pkl_path}")
    _model = joblib.load(pkl_path)
    logger.info(
        f"[init] Loaded {type(_model).__name__}  "
        f"n_estimators={_model.n_estimators}  "
        f"classes={_model.classes_.tolist()}"
    )


# ── Validation helper ─────────────────────────────────────────────────────────

def _parse_v2_tensor(inp: dict, name: str, n_features: int) -> np.ndarray:
    """Validate and return a float32 numpy array from a V2 tensor descriptor.

    Raises ValueError for:
      - Wrong tensor name
      - Missing 'shape' or 'data' keys
      - data length != product of shape dimensions
      - batch size > _MAX_BATCH
      - unexpected number of features in last dimension
      - NaN or Inf values
    """
    if inp.get("name") != name:
        raise ValueError(f"Expected tensor '{name}', got '{inp.get('name')}'")

    shape = inp.get("shape")
    data  = inp.get("data")
    if shape is None or data is None:
        raise ValueError(f"Tensor '{name}' is missing 'shape' or 'data'")

    # Flatten nested lists so we can check the total element count
    flat = [v for row in data for v in row] if data and isinstance(data[0], list) else list(data)
    expected_len = 1
    for d in shape:
        expected_len *= d
    if len(flat) != expected_len:
        raise ValueError(
            f"Tensor '{name}': shape {shape} implies {expected_len} elements "
            f"but data has {len(flat)}"
        )

    if shape[0] > _MAX_BATCH:
        raise ValueError(
            f"Batch size {shape[0]} exceeds maximum allowed ({_MAX_BATCH})"
        )

    if shape[-1] != n_features:
        raise ValueError(
            f"Tensor '{name}': expected {n_features} features in last dimension, "
            f"got {shape[-1]}"
        )

    arr = np.array(flat, dtype=np.float32).reshape(shape)
    if not np.isfinite(arr).all():
        raise ValueError(f"Tensor '{name}' contains NaN or Inf values")
    return arr


# ── Response helpers ──────────────────────────────────────────────────────────

def _ok(body: dict) -> "AMLResponse":
    # Note: AMLResponse in sklearn-1.5 env does not accept 'headers' kwarg
    return AMLResponse(json.dumps(body), 200)


def _err(msg: str, status: int = 400) -> "AMLResponse":
    return AMLResponse(json.dumps({"error": msg}), status)


# ── Scoring entry point ───────────────────────────────────────────────────────

@rawhttp
def run(request: "AMLRequest") -> "AMLResponse":
    """
    AML rawhttp scoring entry point — KFServing V2 inference for iris_classifier.

    Accepts KFServing V2 inference JSON and returns a V2-compatible response.
    Proper HTTP status codes are returned on all error paths (4xx / 5xx).

    Example request body:
    {
      "inputs": [{"name": "float_input", "shape": [1, 4],
                  "datatype": "FP32", "data": [[5.1, 3.5, 1.4, 0.2]]}]
    }
    """
    req_id = str(uuid.uuid4())[:8]
    t0     = time.perf_counter()
    status = 200

    try:
        if (getattr(request, "method", "POST") or "POST").upper() != "POST":
            status = 405
            return _err("Method not allowed", status)

        # ── Parse body ───────────────────────────────────────────────────────
        try:
            raw = (
                request.get_data(as_text=True) if hasattr(request, "get_data")
                else (request.data.decode() if isinstance(request.data, bytes)
                      else request.data)
            )
            payload = json.loads(raw) if raw else {}
        except Exception as exc:
            status = 400
            return _err(f"Could not parse JSON body: {exc}", status)

        # ── Locate and validate input tensor ─────────────────────────────────
        inp = next(
            (i for i in payload.get("inputs", []) if i.get("name") == "float_input"),
            None,
        )
        if inp is None:
            status = 400
            return _err("Required input tensor 'float_input' not found", status)

        try:
            X = _parse_v2_tensor(inp, "float_input", _N_FEATURES)
        except ValueError as exc:
            status = 400
            return _err(str(exc), status)

        # ── Inference ─────────────────────────────────────────────────────────
        labels = _model.predict(X).astype(np.int64).tolist()
        probas = _model.predict_proba(X).astype(np.float32).tolist()
        bs, nc = len(labels), len(probas[0])

        return _ok({
            "model_name":    MODEL_NAME,
            "model_version": "1",
            "outputs": [
                {"name": "label",         "shape": [bs],     "datatype": "INT64",
                 "data": labels},
                {"name": "probabilities", "shape": [bs, nc], "datatype": "FP32",
                 "data": [p for row in probas for p in row]},
            ],
        })

    except Exception as exc:
        status = 500
        logger.error(f"[run] req_id={req_id} Unhandled error: {exc}", exc_info=True)
        return _err(str(exc), status)

    finally:
        # Structured log line — queryable in Azure Monitor / Log Analytics
        logger.info(json.dumps({
            "req_id":     req_id,
            "model":      MODEL_NAME,
            "status":     status,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }))
