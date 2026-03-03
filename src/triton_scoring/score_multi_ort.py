"""
Multi-model KFServing V2 scoring script for AML Kubernetes Online Endpoints.

Serves two models from a single endpoint with distinct per-model URLs.
Implements Triton Inference Server's V2 inference protocol on CPU —
no CUDA dependency, no GPU driver required.

Models served
─────────────
  iris_classifier  — sklearn RandomForestClassifier (Iris dataset)
                     loaded from:  {model_dir}/iris_classifier/1/model.pkl
  pytorch_sine     — PyTorch MLP sine-wave regressor
                     loaded from:  {model_dir}/pytorch_sine/1/model_params.npz
                     (pure-numpy forward pass; no onnxruntime dependency)

Routing — three mechanisms, tried in order
─────────────────────────────────────────
  1. Triton-style path routing  POST {scoring_uri}/v2/models/{model_name}/infer
  2. Query-parameter routing    POST {scoring_uri}?model={model_name}
  3. Body-field routing         POST {scoring_uri}  {"model_name": "...", "inputs": [...]}

Additional V2 endpoints
───────────────────────
  GET  {scoring_uri}/v2/health/ready               server health
  GET  {scoring_uri}/v2/models/{model_name}/ready  per-model readiness
  GET  {scoring_uri}/v2/repository/index           list loaded models
  POST {scoring_uri}/v2/repository/models/{model}/load    load a model
  POST {scoring_uri}/v2/repository/models/{model}/unload  unload a model
  (model control mode — enables dynamic model management at runtime)
"""
import glob
import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid

import joblib
import numpy as np

# ── AML rawhttp decorator ────────────────────────────────────────────────────
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

_MAX_BATCH = 1024   # guard against unreasonably large request payloads

# ── Routing regexes ──────────────────────────────────────────────────────────
_V2_INFER_RE   = re.compile(r"/v2/models/([^/?\s]+)/infer")
_V2_READY_RE   = re.compile(r"/v2/models/([^/?\s]+)/ready")
_V2_LOAD_RE    = re.compile(r"/v2/repository/models/([^/?\s]+)/load")
_V2_UNLOAD_RE  = re.compile(r"/v2/repository/models/([^/?\s]+)/unload")
_V2_INDEX_RE   = re.compile(r"/v2/repository/index")
_V2_HEALTH_RE  = re.compile(r"/v2/health")

# ── Global model registry ────────────────────────────────────────────────────
# Maps model_name → {"type": ..., ...model-specific fields...}
_registry: dict = {}
_model_dir: str = ""

# ── Cross-worker registry synchronisation ────────────────────────────────────
# Gunicorn spawns N worker *processes* — each has its own memory space and its
# own copy of _registry.  A load/unload request handled by worker A would not
# normally affect workers B, C, D.
#
# Fix: whenever the registry changes, atomically write the desired model list
# to a shared temp file (_STATE_FILE).  Every worker checks that file's mtime
# at the top of run() and syncs its in-process registry if it is stale.
#
# Properties:
#   • No external dependencies (only os.path.getmtime + json + os.replace)
#   • Fast path (no change): a single getmtime() syscall, no lock
#   • Eventually consistent: each worker syncs on its *next* request
#   • Thread-safe within a worker: _registry_lock protects all mutations

_registry_lock = threading.Lock()
_STATE_FILE    = os.path.join(tempfile.gettempdir(), "aml_score_multi_registry.json")
_state_mtime: float = 0.0   # per-process; tracks last-seen mtime of _STATE_FILE


# ═══════════════════════════════════════════════════════════════════════════
# Validation helper
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# Cross-worker state helpers
# ═══════════════════════════════════════════════════════════════════════════

def _save_state() -> None:
    """Atomically persist the list of currently loaded model names.

    Must be called while holding _registry_lock.
    Uses os.replace() for an atomic rename — other workers see either the
    old file or the new one, never a partially-written file.
    """
    state = {"loaded": sorted(_registry.keys())}
    tmp = _STATE_FILE + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, _STATE_FILE)
    logger.info(f"[state] pid={os.getpid()} wrote registry: {state['loaded']}")


def _reload_if_stale() -> None:
    """Sync this worker's _registry from the shared state file if it has changed.

    Fast path (no change): a single os.path.getmtime() syscall, no lock.
    Slow path (change detected): acquires _registry_lock, re-checks under
    lock, then loads/unloads models to match the file's desired list.
    """
    global _state_mtime
    try:
        mtime = os.path.getmtime(_STATE_FILE)
    except FileNotFoundError:
        return  # file not yet written (before the first init() completes)

    if mtime <= _state_mtime:
        return  # nothing changed — fast path

    with _registry_lock:
        # Re-read mtime under lock: another thread in this worker may have
        # already synced while we were waiting to acquire the lock.
        try:
            mtime = os.path.getmtime(_STATE_FILE)
        except FileNotFoundError:
            return
        if mtime <= _state_mtime:
            return

        try:
            with open(_STATE_FILE) as f:
                desired = set(json.load(f).get("loaded", []))
        except Exception as exc:
            logger.error(f"[state] Could not read state file: {exc}")
            return

        loaders = {"iris_classifier": _load_iris, "pytorch_sine": _load_pytorch_sine}
        current = set(_registry.keys())

        for name in current - desired:
            del _registry[name]
            logger.info(f"[state] pid={os.getpid()} unloaded '{name}' (synced)")

        for name in desired - current:
            if name in loaders:
                try:
                    _registry[name] = loaders[name](_model_dir)
                    logger.info(f"[state] pid={os.getpid()} loaded '{name}' (synced)")
                except Exception as exc:
                    logger.error(f"[state] pid={os.getpid()} failed to load '{name}': {exc}")

        _state_mtime = mtime
        logger.info(f"[state] pid={os.getpid()} registry now: {sorted(_registry.keys())}")


# ═══════════════════════════════════════════════════════════════════════════
# Model loading helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_iris(model_dir: str) -> dict:
    """Load sklearn iris_classifier from model.pkl."""
    pattern = os.path.join(model_dir, "**", "iris_classifier", "**", "model.pkl")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        # broader search
        candidates = glob.glob(os.path.join(model_dir, "**", "model.pkl"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"model.pkl not found under {model_dir!r}")
    path = candidates[0]
    sk_model = joblib.load(path)
    logger.info(f"[iris_classifier] loaded sklearn model from {path}")
    return {"type": "sklearn", "model": sk_model, "path": path}


def _load_pytorch_sine(model_dir: str) -> dict:
    """Load PyTorch sine MLP weights from model_params.npz for numpy inference."""
    pattern = os.path.join(model_dir, "**", "pytorch_sine", "**", "model_params.npz")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(f"model_params.npz not found under {model_dir!r}")
    path = candidates[0]
    params = np.load(path)
    entry = {
        "type": "numpy_mlp",
        "path": path,
        "W1": params["W1"], "b1": params["b1"],
        "W2": params["W2"], "b2": params["b2"],
        "W3": params["W3"], "b3": params["b3"],
    }
    logger.info(f"[pytorch_sine] loaded MLP weights from {path}  "
                f"W1={entry['W1'].shape}  W2={entry['W2'].shape}  W3={entry['W3'].shape}")
    return entry


# ═══════════════════════════════════════════════════════════════════════════
# AML lifecycle hooks
# ═══════════════════════════════════════════════════════════════════════════

def init():
    """Called once at container start by the AML inference server."""
    global _model_dir
    _model_dir = os.environ.get("AZUREML_MODEL_DIR", "")
    logger.info(f"[init] AZUREML_MODEL_DIR = {_model_dir!r}")
    logger.info(f"[init] rawhttp source     = {_RAWHTTP_SOURCE!r}")
    logger.info(f"[init] pid                = {os.getpid()}")

    for name, loader in [("iris_classifier", _load_iris),
                          ("pytorch_sine",    _load_pytorch_sine)]:
        try:
            _registry[name] = loader(_model_dir)
            logger.info(f"[init] {name} — loaded OK")
        except Exception as exc:
            logger.error(f"[init] {name} — FAILED: {exc}", exc_info=True)

    # Write initial shared state so workers started later can sync from it.
    with _registry_lock:
        _save_state()

    logger.info(f"[init] Models ready: {sorted(_registry)}")


# ═══════════════════════════════════════════════════════════════════════════
# Per-model inference implementations
# ═══════════════════════════════════════════════════════════════════════════

def _infer_iris(payload: dict) -> dict:
    """Run iris_classifier sklearn inference."""
    sk = _registry["iris_classifier"]["model"]
    inp = next((i for i in payload.get("inputs", []) if i.get("name") == "float_input"), None)
    if inp is None:
        raise ValueError("Required input tensor 'float_input' not found in request")
    X = _parse_v2_tensor(inp, "float_input", n_features=4)

    labels = sk.predict(X).astype(np.int64).tolist()
    probas = sk.predict_proba(X).astype(np.float32).tolist()
    bs, nc = len(labels), len(probas[0])
    return {
        "model_name": "iris_classifier",
        "model_version": "1",
        "outputs": [
            {"name": "label",         "shape": [bs],     "datatype": "INT64",
             "data": labels},
            {"name": "probabilities", "shape": [bs, nc], "datatype": "FP32",
             "data": [p for row in probas for p in row]},
        ],
    }


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _infer_pytorch_sine(payload: dict) -> dict:
    """Run pytorch_sine MLP via pure-numpy forward pass."""
    e = _registry["pytorch_sine"]
    inp = next((i for i in payload.get("inputs", []) if i.get("name") == "x"), None)
    if inp is None:
        raise ValueError("Required input tensor 'x' not found in request")
    X = _parse_v2_tensor(inp, "x", n_features=1)

    # Forward pass: MLP(1 → hidden → hidden → 1) with ReLU activations
    h = _relu(X @ e["W1"].T + e["b1"])
    h = _relu(h @ e["W2"].T + e["b2"])
    y = (h @ e["W3"].T + e["b3"])
    return {
        "model_name": "pytorch_sine",
        "model_version": "1",
        "outputs": [
            {"name": "y", "shape": list(y.shape), "datatype": "FP32",
             "data": y.flatten().tolist()},
        ],
    }


_HANDLERS = {
    "iris_classifier": _infer_iris,
    "pytorch_sine":    _infer_pytorch_sine,
}


# ═══════════════════════════════════════════════════════════════════════════
# Response helpers
# ═══════════════════════════════════════════════════════════════════════════

def _ok(body: dict) -> "AMLResponse":
    # Note: AMLResponse in sklearn-1.5 env does not accept 'headers' kwarg
    return AMLResponse(json.dumps(body), 200)


def _err(msg: str, status: int = 400) -> "AMLResponse":
    return AMLResponse(json.dumps({"error": msg}), status)


def _dispatch(model_name: str, payload: dict, req_id: str = "") -> "AMLResponse":
    """Route inference request to the correct model handler."""
    t0 = time.perf_counter()
    if model_name not in _registry:
        logger.info(json.dumps({
            "req_id": req_id, "model": model_name,
            "status": 404, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }))
        return _err(
            f"Model '{model_name}' is not loaded. "
            f"Available: {sorted(_registry)}",
            status=404,
        )
    try:
        result = _HANDLERS[model_name](payload)
        logger.info(json.dumps({
            "req_id": req_id, "model": model_name,
            "status": 200, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }))
        return _ok(result)
    except Exception as exc:
        logger.error(f"[{model_name}] req_id={req_id} Inference error: {exc}", exc_info=True)
        logger.info(json.dumps({
            "req_id": req_id, "model": model_name,
            "status": 500, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }))
        return _err(str(exc), status=500)


# ═══════════════════════════════════════════════════════════════════════════
# rawhttp entry point — handles ALL requests to the scoring endpoint
# ═══════════════════════════════════════════════════════════════════════════

@rawhttp
def run(request: "AMLRequest") -> "AMLResponse":
    """
    AML rawhttp scoring entry point.

    Routing priority:
      1. Triton-style path  POST /v2/models/{model_name}/infer
      2. Query parameter    POST ?model={model_name}
      3. Body field         {"model_name": "...", "inputs": [...]}

    Also handles V2 management endpoints (health, readiness, model control).
    """
    # Sync registry from shared state file if another worker changed it
    _reload_if_stale()

    # Extract request metadata safely
    req_id = str(uuid.uuid4())[:8]
    path   = getattr(request, "full_path", None) or getattr(request, "path", "") or ""
    method = (getattr(request, "method", "POST") or "POST").upper()

    # Query params — request.args is Flask's ImmutableMultiDict
    try:
        args = dict(request.args) if hasattr(request, "args") else {}
    except Exception:
        args = {}

    logger.info(f"[run] req_id={req_id} {method} {path}")

    # ── Server health ────────────────────────────────────────────────────
    if _V2_HEALTH_RE.search(path):
        return _ok({"status": "ready", "models": sorted(_registry)})

    # ── Repository index ─────────────────────────────────────────────────
    if _V2_INDEX_RE.search(path) and method == "GET":
        return _ok({
            "models": [
                {"name": n, "state": "READY", "version": "1"}
                for n in sorted(_registry)
            ]
        })

    # ── Model readiness ──────────────────────────────────────────────────
    m = _V2_READY_RE.search(path)
    if m:
        mn = m.group(1)
        if mn in _registry:
            return _ok({"model": mn, "ready": True})
        return _err(f"Model '{mn}' not loaded", 404)

    # ── Model control: load ──────────────────────────────────────────────
    m = _V2_LOAD_RE.search(path)
    if m and method == "POST":
        mn = m.group(1)
        loaders = {"iris_classifier": _load_iris, "pytorch_sine": _load_pytorch_sine}
        if mn not in loaders:
            return _err(f"Unknown model '{mn}'", 404)
        try:
            with _registry_lock:
                _registry[mn] = loaders[mn](_model_dir)
                _save_state()   # propagate change to all other workers
            return _ok({"model": mn, "state": "READY"})
        except Exception as exc:
            return _err(str(exc), 500)

    # ── Model control: unload ────────────────────────────────────────────
    m = _V2_UNLOAD_RE.search(path)
    if m and method == "POST":
        mn = m.group(1)
        with _registry_lock:
            if mn not in _registry:
                return _err(f"Model '{mn}' not loaded", 404)
            del _registry[mn]
            _save_state()   # propagate change to all other workers
        return _ok({"model": mn, "state": "UNLOADED"})

    # ── Inference (POST only) ────────────────────────────────────────────
    if method != "POST":
        return _err("Method not allowed", 405)

    # Parse request body
    try:
        raw = request.get_data(as_text=True) if hasattr(request, "get_data") else (
            request.data.decode() if isinstance(request.data, bytes) else request.data
        )
        payload = json.loads(raw) if raw else {}
    except Exception as exc:
        return _err(f"Could not parse JSON body: {exc}", 400)

    # Priority 1: Triton-style path  /v2/models/{model_name}/infer
    m = _V2_INFER_RE.search(path)
    if m:
        return _dispatch(m.group(1), payload, req_id)

    # Priority 2: query parameter  ?model={model_name}
    model_name = args.get("model", [""])[0] if isinstance(args.get("model"), list) \
                 else args.get("model", "")
    if model_name:
        return _dispatch(model_name, payload, req_id)

    # Priority 3: body field  {"model_name": "..."}
    model_name = payload.get("model_name", "")
    if model_name:
        return _dispatch(model_name, payload, req_id)

    # No routing information — return helpful error
    return _err(
        "Cannot determine target model. Use one of:\n"
        "  (1) path routing:  POST {uri}/v2/models/{model_name}/infer\n"
        "  (2) query param:   POST {uri}?model={model_name}\n"
        "  (3) body field:    {\"model_name\": \"...\", \"inputs\": [...]}\n"
        f"  Available models: {sorted(_registry)}",
        status=400,
    )
