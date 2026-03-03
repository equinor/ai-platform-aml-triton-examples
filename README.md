# AI Platform AML Triton Examples

End-to-end examples for training models on **Azure ML (AML)** using pipelines,
converting them to ONNX / saving inference weights, and deploying to **AKS
Kubernetes Online Endpoints** with a Triton-compatible KFServing V2 API.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`train-on-ai-platform-aks-triton.ipynb`](#single-model-notebook) | Train one scikit-learn Iris classifier, deploy it as a single-model KFServing V2 endpoint |
| [`train-on-ai-platform-aks-triton-n-models.ipynb`](#multi-model-notebook) | Train sklearn + PyTorch models, deploy **both** to a single endpoint with model control mode |

---

## Single-Model Notebook

`train-on-ai-platform-aks-triton.ipynb`

### What it does

| Step | Description |
|------|-------------|
| 1 | Install Python dependencies |
| 2 | Run a 3-stage AML pipeline: **train → analyse → batch-score** (Iris dataset) |
| 3 | Download `model.pkl` from the training child run |
| 4 | Convert `RandomForestClassifier` → ONNX with `skl2onnx` |
| 5 | Build a Triton model repository (`config.pbtxt` + `model.onnx` + `model.pkl`) |
| 6 | Register the model repository in Azure ML as a `triton_model` asset |
| 7 | Create a `KubernetesOnlineEndpoint` on the `cpu-2` compute pool |
| 8 | Deploy `score_ort.py` (KFServing V2 via sklearn — zero CUDA dependency) |
| 9 | Test the endpoint with a KFServing V2 inference request |

### Inference API

**Request:**
```json
{
  "inputs": [{
    "name": "float_input",
    "shape": [1, 4],
    "datatype": "FP32",
    "data": [[5.1, 3.5, 1.4, 0.2]]
  }],
  "outputs": [{"name": "label"}, {"name": "probabilities"}]
}
```

**Response:**
```json
{
  "model_name": "iris_classifier",
  "model_version": "1",
  "outputs": [
    {"name": "label",         "shape": [1],    "datatype": "INT64", "data": [0]},
    {"name": "probabilities", "shape": [1, 3], "datatype": "FP32",  "data": [1.0, 0.0, 0.0]}
  ]
}
```

Class mapping: `0 = setosa`, `1 = versicolor`, `2 = virginica`

---

## Multi-Model Notebook

`train-on-ai-platform-aks-triton-n-models.ipynb`

### What it does

| Step | Description |
|------|-------------|
| 1–6 | Same as single-model notebook (Iris classifier via AML pipeline → ONNX) |
| 7 | Train a **PyTorch sine-wave MLP** locally (3-layer, 1 → 32 → 32 → 1) |
| 8 | Export PyTorch model to ONNX **and** save weights as `model_params.npz` for CPU serving |
| 9 | Build a multi-model Triton repository with `iris_classifier/` and `pytorch_sine/` |
| 10 | Register the multi-model repository in Azure ML |
| 11 | Create a `KubernetesOnlineEndpoint` on `cpu-2` |
| 12 | Deploy `score_multi_ort.py` — serves **both models** with model control mode |
| 13 | Test each model via **query-parameter routing** |

### PyTorch SineMLP Architecture

A small 3-layer MLP trained to approximate sin(x) over `[-π, π]`:

```
Input(1) → Linear(32) → ReLU → Linear(32) → ReLU → Linear(1)
```

Trained for 3 000 epochs with Adam optimiser (MSE loss < 0.003). Served
at inference time using a **pure-numpy forward pass** — no onnxruntime or
torch dependency on the endpoint container.

### Distinct URLs per Model

Despite both models being hosted in a single endpoint, each model gets a
distinct URL via query-parameter routing:

| Model | URL |
|-------|-----|
| `iris_classifier` | `{scoring_uri}?model=iris_classifier` |
| `pytorch_sine`    | `{scoring_uri}?model=pytorch_sine` |

> **AML Gateway Routing Note**: The AML Kubernetes gateway only forwards
> requests to the `/score` path. Sub-path routing (e.g.
> `/score/v2/models/{name}/infer`) returns HTTP 404 from the gateway before
> reaching the scoring script. Query-parameter routing (`?model={name}`) is
> fully forwarded and is the recommended pattern.

### Model Control Mode

`score_multi_ort.py` implements the Triton V2 model management API.
These endpoints are handled by the scoring script but are only accessible
from within the container / sidecar (the AML gateway blocks sub-paths from
outside):

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/v2/health/ready` | Server readiness |
| `GET`  | `/v2/models/{name}/ready` | Per-model readiness |
| `POST` | `/v2/repository/models/{name}/load` | Load a model |
| `POST` | `/v2/repository/models/{name}/unload` | Unload a model |
| `GET`  | `/v2/repository/index` | List all registered models |

### Inference APIs

**iris_classifier — Iris flower classification:**

```bash
curl -k -X POST '{scoring_uri}?model=iris_classifier' \
  -H 'Authorization: Bearer $API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"float_input","shape":[1,4],"datatype":"FP32","data":[[5.1,3.5,1.4,0.2]]}]}'
```

Response:
```json
{
  "model_name": "iris_classifier", "model_version": "1",
  "outputs": [
    {"name": "label",         "shape": [1],    "datatype": "INT64", "data": [0]},
    {"name": "probabilities", "shape": [1, 3], "datatype": "FP32",  "data": [1.0, 0.0, 0.0]}
  ]
}
```

**pytorch_sine — Sine-wave regression:**

```bash
curl -k -X POST '{scoring_uri}?model=pytorch_sine' \
  -H 'Authorization: Bearer $API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"x","shape":[1,1],"datatype":"FP32","data":[[1.5708]]}]}'
```

Response:
```json
{
  "model_name": "pytorch_sine", "model_version": "1",
  "outputs": [{"name": "y", "shape": [1, 1], "datatype": "FP32", "data": [1.0026]}]
}
```

---

## Repository Structure

```
ai-platform-aml-triton-examples/
├── train-on-ai-platform-aks-triton.ipynb          # Single-model notebook
├── train-on-ai-platform-aks-triton-n-models.ipynb # Multi-model notebook
├── requirements.txt                                # Python dependencies
├── .gitignore
├── README.md
├── artifacts/                                      # Runtime outputs (gitignored)
│   ├── model_download/                             #   model.pkl downloaded from AML
│   └── triton_model_repo/                          #   Triton model repo built locally
└── src/
    ├── _helpers/
    │   └── load_tags.py                            # Reads Kubernetes ConfigMap tags
    ├── iris_pipeline/
    │   ├── train.py                                # Pipeline step 1: train + save model.pkl
    │   ├── analysis.py                             # Pipeline step 2: evaluate + write report
    │   └── score.py                                # Pipeline step 3: batch inference
    └── triton_scoring/
        ├── score_ort.py                            # Single-model CPU endpoint (rawhttp, KFServing V2)
        ├── score_multi_ort.py                      # Multi-model CPU endpoint (model control + cross-worker sync)
        └── score.py                                # GPU endpoint: Triton subprocess proxy
```

---

## Prerequisites

- Python 3.9+
- Access to an **Azure ML workspace** with:
  - A `cpu-2` Kubernetes compute target (or another listed target)
  - The `sklearn-1.5` curated environment (available in the AzureML registry)
  - Managed Identity configured on the Jupyter pod (`AZURE_CLIENT_ID` env var set)
- Jupyter running **inside the AKS cluster** so that `load_tags.py` can read the
  Kubernetes ConfigMap (or replace `load_tags` with hardcoded values — see below)

---

## Quick Start

### 1. Install dependencies

```bash
cd ai-platform-aml-triton-examples
pip install -r requirements.txt
```

### 2. Start Jupyter from the project root

```bash
# Always start Jupyter from within this directory so relative paths work
jupyter lab
```

### 3. Open and run a notebook

- **Single model**: open `train-on-ai-platform-aks-triton.ipynb`
- **Multiple models**: open `train-on-ai-platform-aks-triton-n-models.ipynb`

Run all cells top-to-bottom.

---

## Configuration

Edit **Cell 7** in either notebook to set:

```python
environment      = "azureml://registries/azureml/environments/sklearn-1.5/versions/26"
compute_target   = "cpu-2"          # Change to a GPU target for native Triton

tags = {
    "Purpose":   "Project Resources",
    "by_person": "your-name",       # Set your name here
}
```

### Available Compute Targets

| Compute | Time Slicing | CPU | RAM |
|---------|-------------|-----|-----|
| `cpu-2` / `cpu-4` | 2×, 4× | 14 vCPU | 46 Gi |
| `gput41-2` / `gput41-4` | 2×, 4× | 6 vCPU | 44 Gi |
| `gpuv1001-2` / `gpuv1001-4` | 2×, 4× | 4 vCPU | 98 Gi |
| `gpua100-2` / `gpua100-4` | 2×, 4× | 22 vCPU | 202 Gi |

---

## CPU vs GPU Deployment

| Scenario | Environment | Scoring Script | Compute |
|----------|-------------|----------------|---------|
| **CPU (default)** | `sklearn-1.5` (curated) | `score_ort.py` / `score_multi_ort.py` | `cpu-2` |
| GPU (native Triton) | `tritonNcdEnv:23.08.02-py3` | `score.py` | `gput41-2` etc. |

> **Why not native Triton on CPU?**
> `nvcr/nvidia/tritonserver` images are compiled against CUDA and require NVIDIA
> GPU drivers on the host. CPU-only AKS nodes have no NVIDIA driver, causing
> `tritonserver` to crash with `CrashLoopBackOff`.
> `score_ort.py` / `score_multi_ort.py` expose the **identical KFServing V2 API**
> without any CUDA dependency.

---

## Running Without Kubernetes ConfigMap (Local / Dev)

If you are running outside the AKS cluster (e.g., locally), the `load_tags` call
will fail. Replace the `load_tags` cell with hardcoded values:

```python
subscription_id  = "your-subscription-id"
aml_workspace_rg = "your-resource-group"
aml_workspace    = "your-workspace-name"
tags = {"Purpose": "Project Resources", "by_person": "your-name"}
```

---

## Triton Model Repository Layout

### Single-model
```
artifacts/triton_model_repo/
└── iris_classifier/
    ├── config.pbtxt        # backend: "onnxruntime", KIND_CPU
    └── 1/
        ├── model.onnx      # ONNX export (for GPU nodes with native Triton)
        └── model.pkl       # sklearn pickle (for score_ort.py on CPU nodes)
```

### Multi-model
```
artifacts/triton_model_repo/
├── iris_classifier/
│   ├── config.pbtxt
│   └── 1/
│       ├── model.onnx
│       └── model.pkl
└── pytorch_sine/
    ├── config.pbtxt
    └── 1/
        ├── model.onnx         # ONNX export (for GPU nodes with native Triton)
        └── model_params.npz   # W1,b1,W2,b2,W3,b3 for pure-numpy CPU serving
```

---

## Reliability & Production Hardening

The scoring scripts include several hardening measures beyond a minimal proof-of-concept.

### Input Validation

Both `score_ort.py` and `score_multi_ort.py` validate every incoming tensor through
`_parse_v2_tensor()` before calling the model:

| Check | Detail |
|-------|--------|
| Tensor name | Must match the expected name (`float_input`, `x`, etc.) |
| Shape × data consistency | `product(shape)` must equal `len(data)` |
| Batch size cap | Rejects batches > 1 024 rows (configurable via `_MAX_BATCH`) |
| Feature count | Last dimension must equal the expected number of features |
| Finite values | Rejects payloads containing `NaN` or `Inf` |

Validation failures return **HTTP 400** with a JSON `{"error": "..."}` body.

### HTTP Status Codes

`score_ort.py` uses the `@rawhttp` decorator so the scoring function receives the
full `AMLRequest` and returns an `AMLResponse` with the correct status code:

| Condition | Status |
|-----------|--------|
| Method is not POST | 405 |
| Malformed JSON body | 400 |
| Missing / invalid input tensor | 400 |
| Unrecognised model name | 404 |
| Unhandled server error | 500 |
| Success | 200 |

### Structured Request Logging

Every request in both scoring scripts emits a single JSON log line to the AML
inference server log (queryable via Azure Monitor / Log Analytics):

```json
{"req_id": "a1b2c3d4", "model": "iris_classifier", "status": 200, "latency_ms": 3.7}
```

| Field | Description |
|-------|-------------|
| `req_id` | First 8 chars of a UUID — correlates log lines to a single request |
| `model` | Model name that handled the request |
| `status` | HTTP status code returned to the caller |
| `latency_ms` | Wall-clock time from request receipt to response, in milliseconds |

### Cross-Worker Registry Sync (`score_multi_ort.py`)

gunicorn spins up multiple worker processes; each has its own memory space, so a
load/unload via one worker would normally be invisible to the others.
`score_multi_ort.py` uses a file-based, eventually-consistent approach that requires
no external services (no Redis, no database):

- **Write path** — after every `load` or `unload`, the authoritative worker atomically
  writes `{"loaded": [...]}` to a shared temp file via `os.replace()` (rename is
  atomic on POSIX).
- **Read path** — at the start of every `run()` call, each worker compares the file's
  `mtime` to its last-seen mtime.  If unchanged, the check costs a single `getmtime()`
  syscall.  If changed, the worker acquires `_registry_lock` and re-syncs.
- **Thread safety** — a `threading.Lock` serialises all registry mutations within a
  single worker process.

This means a load/unload propagates to all workers within one request cycle (typically
< 100 ms) without any explicit inter-process communication.

### Deployment Timeouts

Long-running Azure SDK operations are guarded by explicit timeouts so notebooks do
not hang indefinitely on transient Azure API issues:

| Operation | Timeout |
|-----------|---------|
| AML pipeline `jobs.stream()` | 2 hours (daemon thread with `.join(timeout=7200)`) |
| Endpoint create / update `.result()` | 10 minutes |
| Deployment create / update `.result()` | 30 minutes |
| Traffic update `.result()` | 5 minutes |

---

## Key Design Decisions

- **ONNX Runtime backend** — used instead of FIL (Forest Inference Library) because
  FIL is not compiled into the standard `tritonserver:23.08-py3` image.
- **`score_ort.py` / `score_multi_ort.py` on CPU** — avoids the CUDA dependency of
  native `tritonserver` while maintaining full KFServing V2 API compatibility.
- **`@rawhttp` decorator in `score_ort.py`** — exposes the raw `AMLRequest` so the
  function can inspect the HTTP method and return proper 4xx / 5xx `AMLResponse`
  objects instead of always returning HTTP 200.
- **`_parse_v2_tensor()` validation helper** — centralised, reusable V2 tensor
  validator; raises `ValueError` with a descriptive message that is forwarded to the
  caller as an HTTP 400 body.
- **File-based cross-worker registry sync** — stdlib-only (`os.replace`, `getmtime`,
  `threading.Lock`); no external dependency required for eventually-consistent model
  control across gunicorn workers.
- **`skl2onnx` with `zipmap=False`** — ensures probability output is a plain
  `float32` array rather than a list-of-dicts, required for Triton ONNX Runtime's
  static output shape declaration.
- **Pure-numpy MLP forward pass** — pytorch_sine weights are saved as `.npz`
  (W1,b1,W2,b2,W3,b3) and run as numpy matmuls at inference time; no torch or
  onnxruntime required in the serving container.
- **`dynamo=False` for ONNX export** — `torch>=2.0` defaults to the new `dynamo`
  exporter which requires `onnxscript`. Using the legacy TorchScript exporter
  avoids this extra dependency.
- **Query-parameter routing** — AML Kubernetes gateway only forwards `/score`;
  sub-paths return 404. Query params (`?model=X`) are forwarded and provide
  distinct URLs per model within a single endpoint.
- **`AMLResponse(body, status_code)` only** — the `AMLResponse` class in the
  `sklearn-1.5` curated environment does not accept a `headers` keyword argument.
- **`opset=12`** — broad compatibility with ONNX Runtime 1.15/1.16 bundled in the
  Triton 23.08 image.
