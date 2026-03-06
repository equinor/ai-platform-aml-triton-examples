# AI Platform AML Triton Examples

End-to-end examples for training models on **Azure ML (AML)** using pipelines,
converting them to ONNX / saving inference weights, and deploying to **AKS
Kubernetes Online Endpoints** with a Triton-compatible KFServing V2 API.

---

## Notebooks

| Notebook | Output | Description |
|----------|--------|-------------|
| [`train-on-ai-platform-aks-triton.ipynb`](#single-model-notebook) | [`train-on-ai-platform-aks-triton-output.ipynb`](train-on-ai-platform-aks-triton-output.ipynb) | Train one scikit-learn Iris classifier, deploy it as a single-model KFServing V2 endpoint via AzureML Online Endpoint |
| [`train-on-ai-platform-aks-triton-n-models.ipynb`](#multi-model-notebook) | [`train-on-ai-platform-aks-triton-n-models-output.ipynb`](train-on-ai-platform-aks-triton-n-models-output.ipynb) | Train sklearn + PyTorch models, deploy **both** to a single endpoint with model control mode |
| [`sklearn-triton-kserve-deployment.ipynb`](#kserve-sklearn-notebook) | [`sklearn-triton-kserve-deployment-output.ipynb`](sklearn-triton-kserve-deployment-output.ipynb) | Fetch the latest `iris_classifier` Triton model from AML, deploy directly to **KServe** (no AzureML Online Endpoint, no retraining) |
| [`pytorch-triton-kserve-deployment.ipynb`](#kserve-pytorch-notebook) | [`pytorch-triton-kserve-deployment-output.ipynb`](pytorch-triton-kserve-deployment-output.ipynb) | Fetch the latest `pytorch_sine` Triton model from AML, deploy directly to **KServe** using explicit model control mode |

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

## KServe Sklearn Notebook

`sklearn-triton-kserve-deployment.ipynb` · output: [`sklearn-triton-kserve-deployment-output.ipynb`](sklearn-triton-kserve-deployment-output.ipynb)

Deploys an already-registered Triton model to **KServe** running inside the same AKS cluster —
no pipeline retraining, no AzureML Online Endpoint. The model is served by the native Triton
Inference Server via the KServe `InferenceService` CRD.

### vs. `train-on-ai-platform-aks-triton.ipynb`

| | AzureML Endpoint notebook | KServe notebook |
|-|--------------------------|-----------------|
| **Serving runtime** | `score_ort.py` (sklearn + joblib) | Native Triton Inference Server |
| **Deployment target** | AzureML `KubernetesOnlineEndpoint` | KServe `InferenceService` CRD |
| **Retraining** | Full AML pipeline (train → analyse → batch-score) | None — fetches existing registered model |
| **Auth** | AML API key (`Authorization: Bearer`) | None (cluster-internal) |

### Notebook sections

#### Section 1 — Initial Setup & Configuration
Install dependencies, set configuration variables, import libraries, and connect to the AML
workspace via Managed Identity.

#### Section 2 — Model Preparation

| Step | Description |
|------|-------------|
| 2.1 | Fetch latest `triton_model` from AML registry by name/prefix; fast blob-existence check skips registrations without the required model subdirectory |
| 2.2 | Download locally to verify Triton repository layout (`config.pbtxt`, `1/model.onnx`) and extract metadata |
| 2.3 | Resolve `azureml://` path → `https://` blob URI for the Triton model repository root |
| 2.4 | Retrieve Azure storage account key via Storage Management API (notebook pod has Managed Identity; storage initializer pod does not) |
| 2.5 | Patch `config.pbtxt` (`max_batch_size: 0`, correct explicit tensor dims) and re-upload to Azure Blob |

#### Section 3 — Inference Service Setup, Configuration & Deployment

| Step | Description |
|------|-------------|
| 3.1 | Configure Kubernetes client (in-cluster config when running inside AKS; `az aks get-credentials` fallback) |
| 3.2 | Create `azure-storage-secret` with blob credentials; patch `mlpipeline-minio-artifact.secretkey` with the real Azure key (required because the cluster's `ClusterStorageContainer` hardcodes this secret for `AZURE_STORAGE_ACCESS_KEY` injection) |
| 3.3 | Create KServe `InferenceService` with `V1beta1TritonSpec` and `storageUri`; existing service is deleted first for a clean redeploy |
| 3.4 | Poll until `InferenceService` reaches `Ready = True` (15-second interval, 10-minute timeout) |
| 3.5 | Resolve cluster-internal ClusterIP service URL (external `status.url` is not DNS-resolvable from inside the pod) |

#### Section 4 — Inference Service Testing
Send a KFServing V2 inference request and verify the response. Cleanup cell restores
`mlpipeline-minio-artifact.secretkey` to its original MinIO value.

### Inference API

**Request:**
```json
{
  "inputs": [{"name": "float_input", "shape": [1, 4], "datatype": "FP32",
              "data": [[5.1, 3.5, 1.4, 0.2]]}]
}
```

**Response:**
```json
{
  "model_name": "iris_classifier", "model_version": "1",
  "outputs": [
    {"name": "label",         "shape": [1],    "datatype": "INT64", "data": [0]},
    {"name": "probabilities", "shape": [1, 3], "datatype": "FP32",  "data": [1.0, 0.0, 0.0]}
  ]
}
```

Class mapping: `0 = setosa`, `1 = versicolor`, `2 = virginica`

### Prerequisites

- A `triton_model` registered in AML with an `iris_classifier/` subdirectory
  (run `train-on-ai-platform-aks-triton.ipynb` once to create it)
- KServe installed in the AKS cluster (`kubectl get crds | grep inferenceservice`)
- Notebook running **inside** the AKS cluster (in-cluster K8s config)
- Managed Identity with `Microsoft.Storage/storageAccounts/listkeys/action` on the AML storage account

---

## KServe PyTorch Notebook

`pytorch-triton-kserve-deployment.ipynb` · output: [`pytorch-triton-kserve-deployment-output.ipynb`](pytorch-triton-kserve-deployment-output.ipynb)

Deploys the pre-trained `pytorch_sine` MLP to **KServe** using the native Triton Inference
Server — no retraining, no AzureML Online Endpoint. The model approximates `y = sin(x)`
over `[-π, π]` and was registered by `train-on-ai-platform-aks-triton-n-models.ipynb`.

### vs. `sklearn-triton-kserve-deployment.ipynb`

| | sklearn KServe notebook | PyTorch KServe notebook |
|-|------------------------|------------------------|
| **Model** | `iris_classifier` (RandomForest → ONNX) | `pytorch_sine` (3-layer MLP → ONNX) |
| **Task** | Multi-class classification | Sine-wave regression |
| **config.pbtxt** | Patched (`max_batch_size: 0`, explicit dims) | No patch needed (2D ONNX output matches config) |
| **Model isolation** | Full repo loaded (both models) | `--model-control-mode=explicit --load-model=pytorch_sine` |

### Notebook sections

#### Section 1 — Initial Setup & Configuration
Same package install, config variables, imports, and workspace connection as the sklearn notebook.
`inference_service_name = "pytorch-triton"` and `required_triton_model_name = "pytorch_sine"`.

#### Section 2 — Model Preparation

| Step | Description |
|------|-------------|
| 2.1 | Fetch latest `triton_model` with a `pytorch_sine/` subdirectory from AML registry |
| 2.2 | Download locally; verify `model.onnx` exists and extract input dimension |
| 2.3 | Resolve `https://` blob URI for the Triton model repository root |
| 2.4 | Retrieve Azure storage account key via Storage Management API |
| 2.5 | **Verify** (not patch) `config.pbtxt` — `pytorch_sine` config is compatible as-is |

#### Section 3 — Inference Service Setup, Configuration & Deployment

| Step | Description |
|------|-------------|
| 3.1 | Configure Kubernetes client (in-cluster preferred) |
| 3.2 | Create `azure-storage-secret`; patch `mlpipeline-minio-artifact.secretkey` |
| 3.3 | Deploy `InferenceService` with Triton args `--model-control-mode=explicit --load-model=pytorch_sine` to prevent Triton from attempting to load `iris_classifier` (which has a config incompatibility in the same repo) |
| 3.4 | Poll until `Ready = True` |
| 3.5 | Resolve cluster-internal ClusterIP service URL |

#### Section 4 — Inference Service Testing
Sends three test values (`π/2`, `π`, `0`) and verifies each predicted `y` is within `0.05`
of the true `sin(x)`.

### Inference API

**Request:**
```json
{"inputs": [{"name": "x", "shape": [1, 1], "datatype": "FP32", "data": [[1.5708]]}]}
```

**Response:**
```json
{
  "model_name": "pytorch_sine", "model_version": "1",
  "outputs": [{"name": "y", "shape": [1, 1], "datatype": "FP32", "data": [0.9988]}]
}
```

### Prerequisites

- `train-on-ai-platform-aks-triton-n-models.ipynb` run at least once (registers the multi-model
  Triton repo containing `pytorch_sine/` in AML)
- Same cluster/identity requirements as the sklearn KServe notebook

---

## Repository Structure

```
ai-platform-aml-triton-examples/
├── train-on-ai-platform-aks-triton.ipynb              # Single-model notebook (AzureML endpoint)
├── train-on-ai-platform-aks-triton-n-models.ipynb     # Multi-model notebook (AzureML endpoint)
├── sklearn-triton-kserve-deployment.ipynb             # KServe notebook — iris_classifier (sklearn)
├── sklearn-triton-kserve-deployment-output.ipynb      # Last successful execution output
├── pytorch-triton-kserve-deployment.ipynb             # KServe notebook — pytorch_sine (MLP sine)
├── pytorch-triton-kserve-deployment-output.ipynb      # Last successful execution output
├── requirements.txt                                    # Runtime Python dependencies
├── requirements-dev.txt                            # Dev/test dependencies (pytest, pytest-mock)
├── pytest.ini                                      # Test configuration (testpaths, pythonpath, markers)
├── .gitignore
├── README.md
├── artifacts/                                      # Runtime outputs (gitignored)
│   ├── model_download/                             #   model.pkl downloaded from AML
│   └── triton_model_repo/                          #   Triton model repo built locally
├── src/
│   ├── _helpers/
│   │   └── load_tags.py                            # Reads Kubernetes ConfigMap tags
│   ├── iris_pipeline/
│   │   ├── train.py                                # Pipeline step 1: train + save model.pkl
│   │   ├── analysis.py                             # Pipeline step 2: evaluate + write report
│   │   └── score.py                                # Pipeline step 3: batch inference
│   └── triton_scoring/
│       ├── score_ort.py                            # Single-model CPU endpoint (rawhttp, KFServing V2)
│       ├── score_multi_ort.py                      # Multi-model CPU endpoint (model control + cross-worker sync)
│       └── score.py                                # GPU endpoint: Triton subprocess proxy
└── tests/
    ├── aml_stubs.py                                # AMLRequest / AMLResponse / rawhttp stubs
    ├── conftest.py                                 # Stub injection + shared pytest fixtures
    ├── unit/
    │   ├── test_score_ort.py                       # Unit tests for score_ort.py
    │   ├── test_score_multi_ort.py                 # Unit tests for score_multi_ort.py
    │   └── test_train.py                           # Unit tests for train.py (subprocess)
    └── integration/
        └── test_endpoint_live.py                   # Live-endpoint smoke tests (skipped by default)
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

- **Single model** (train + AzureML endpoint): open `train-on-ai-platform-aks-triton.ipynb`
- **Multiple models** (train + AzureML endpoint): open `train-on-ai-platform-aks-triton-n-models.ipynb`
- **KServe — iris classifier** (no retraining, no AzureML endpoint): open `sklearn-triton-kserve-deployment.ipynb`
- **KServe — pytorch sine** (no retraining, no AzureML endpoint): open `pytorch-triton-kserve-deployment.ipynb`

Run all cells top-to-bottom.

---

## Testing

The repository ships with a self-contained test suite that runs entirely
**without AML credentials, a deployed endpoint, or a GPU**.

### Test structure

```
tests/
├── aml_stubs.py                  # Lightweight AMLRequest / AMLResponse / rawhttp stubs
├── conftest.py                   # Stub injection + shared fixtures (iris_model, model_dir, …)
├── unit/
│   ├── test_score_ort.py         # 24 tests — _parse_v2_tensor, init(), run() for score_ort.py
│   ├── test_score_multi_ort.py   # 27 tests — routing, model control, reload, init for score_multi_ort.py
│   └── test_train.py             #  5 tests — train.py exercised end-to-end via subprocess
└── integration/
    └── test_endpoint_live.py     #  8 tests — smoke tests against a live AML endpoint (skipped by default)
pytest.ini                        # testpaths, pythonpath=src, marker definitions
requirements-dev.txt              # pytest>=7.4, pytest-mock>=3.12
```

### 1. Install dev dependencies

```bash
cd ai-platform-aml-triton-examples
pip install -r requirements-dev.txt
```

### 2. Run the unit tests

```bash
pytest tests/unit/ -v
```

Expected output (all 56 tests passing):

```
tests/unit/test_score_multi_ort.py::TestRelu::test_positive_values_pass_through PASSED
tests/unit/test_score_multi_ort.py::TestInferenceRouting::test_query_param_routes_iris PASSED
...
tests/unit/test_score_ort.py::TestRun::test_returns_200 PASSED
tests/unit/test_train.py::test_accuracy_at_least_90_percent PASSED
...
========================== 56 passed in 7s =====================================
```

### 3. Reading the output

| Symbol | Meaning |
|--------|---------|
| `PASSED` | Test passed |
| `FAILED` | Assertion failed — full diff is printed below the run summary |
| `ERROR` | Unexpected exception during test setup or teardown |
| `SKIPPED` | Test was intentionally skipped (integration tests without env vars set) |

When a test fails, pytest prints the assertion and a contextual diff, for example:

```
FAILED tests/unit/test_score_ort.py::TestRun::test_returns_200
E   assert 400 == 200
E    +  where 400 = <AMLResponse object>.status_code
```

Useful flags for investigating failures:

```bash
# Show full traceback instead of the short summary
pytest tests/unit/ -v --tb=long

# Re-run only the tests that failed in the last run
pytest tests/unit/ -v --last-failed

# Run a specific file, class, or single test
pytest tests/unit/test_score_ort.py -v
pytest tests/unit/test_score_multi_ort.py::TestInferenceRouting -v
pytest tests/unit/test_score_ort.py::TestRun::test_returns_200 -v

# Stop after the first failure
pytest tests/unit/ -v -x
```

### 4. Integration tests (requires a live endpoint)

Integration tests are **skipped automatically** unless both environment variables
are exported in the current shell:

| Variable | Required | Where to find it |
|----------|----------|-----------------|
| `ENDPOINT_URL` | Yes | Azure ML studio → Endpoints → your endpoint → Consume tab → REST endpoint |
| `API_KEY` | Yes | Azure ML studio → Endpoints → your endpoint → Consume tab → Primary key |
| `VERIFY_SSL` | No (default `0`) | Set to `1` to enable TLS certificate verification. Leave unset (or `0`) when the endpoint sits behind a reverse proxy with a private CA certificate (e.g. `unified.aurora.equinor.com`) — equivalent to `curl -k` / `requests.get(..., verify=False)`. |

**What is an SSL certificate and why does it matter here?**

An SSL certificate is a digital ID card that a server presents when you open an
HTTPS connection. Your client checks that the certificate was signed by a trusted
authority (a Certificate Authority, or CA). If the check passes the connection is
encrypted; if it fails the connection is refused.

```
  You (client)                         Server
       │                                  │
       │──── "Hello, who are you?" ──────►│
       │◄─── Certificate ─────────────────│
       │                                  │
       │  Is this signed by a trusted CA? │
       │  ┌──────────────────────────┐    │
       │  │ YES → encrypt & proceed  │    │
       │  │ NO  → connection refused │    │
       │  └──────────────────────────┘    │
```

The `unified.aurora.equinor.com` gateway uses a **private corporate CA** whose
root certificate is not included in Python's default trust store. Setting
`VERIFY_SSL=0` tells the test client to skip the certificate check (equivalent
to `curl -k`) — this is safe because the tests run inside the same private
network as the gateway and the API key still authenticates each request.

Or retrieve them via the Azure CLI:

```bash
export ENDPOINT_URL=$(az ml online-endpoint show \
    -n <endpoint-name> --query scoring_uri -o tsv)
export API_KEY=$(az ml online-endpoint get-credentials \
    -n <endpoint-name> --query primaryKey -o tsv)
```

Then run:

```bash
pytest tests/integration/ -m integration -v
```

Expected output when the endpoint is healthy:

```
tests/integration/test_endpoint_live.py::TestIrisClassifier::test_single_row_returns_200 PASSED
tests/integration/test_endpoint_live.py::TestIrisClassifier::test_predicted_class_is_valid PASSED
tests/integration/test_endpoint_live.py::TestIrisClassifier::test_probabilities_sum_to_one PASSED
tests/integration/test_endpoint_live.py::TestIrisClassifier::test_batch_of_three_rows PASSED
tests/integration/test_endpoint_live.py::TestPytorchSine::test_sine_of_half_pi_approx_one PASSED
tests/integration/test_endpoint_live.py::TestErrorHandling::test_unknown_model_returns_4xx PASSED
tests/integration/test_endpoint_live.py::TestErrorHandling::test_shape_mismatch_returns_error PASSED
tests/integration/test_endpoint_live.py::TestErrorHandling::test_missing_auth_returns_401_or_403 PASSED
========================== 8 passed in 3.2s ====================================
```

Without env vars set, every test is skipped with a clear reason:

```
tests/integration/test_endpoint_live.py::TestIrisClassifier::test_single_row_returns_200
SKIPPED (ENDPOINT_URL and API_KEY environment variables must be set)
...
========================== 8 skipped in 0.1s ===================================
```

> **Note:** `TestPytorchSine::test_sine_of_half_pi_approx_one` targets the
> multi-model endpoint from `train-on-ai-platform-aks-triton-n-models.ipynb`.
> Point `ENDPOINT_URL` at the correct endpoint for the model you want to test.

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

**What is the `@rawhttp` decorator and why is it needed?**

By default the AML inference server unwraps the incoming JSON body into a plain
Python dict and passes it to `run(data)`. The return value is also wrapped and
sent back as **HTTP 200 every time** — even when something went wrong. Callers
have no standard way to tell success from failure.

The `@rawhttp` decorator opts out of that wrapping. The scoring function instead
receives the full HTTP request object (`AMLRequest`) and is responsible for
building and returning an `AMLResponse` with the correct status code.

```
  Default AML mode                    @rawhttp mode
  ────────────────────────────────    ──────────────────────────────────────
  run(data: dict) → dict              run(request: AMLRequest) → AMLResponse
         │                                   │
         │ AML always responds               ├─ wrong method  → 405
         │ with HTTP 200                     ├─ malformed JSON → 400
         ▼                                   ├─ bad tensor     → 400
  HTTP 200 OK                                ├─ server error   → 500
  (even for errors!)                         └─ success        → 200
```

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

> **AKS gateway note:** The Kubernetes Online Endpoint gateway may convert a
> `500` returned by the scoring container into `502 Bad Gateway` before the
> response reaches the caller. Client code and integration tests should treat
> `500` and `502` as equivalent server-error signals.

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

**What is the problem?**

The AML inference container runs **gunicorn**, which forks several worker
processes to handle requests in parallel. Each process has its own isolated
memory. If a `POST /v2/repository/models/iris_classifier/unload` request lands
on Worker A, Worker A removes `iris_classifier` from its in-memory registry —
but Workers B, C, and D still have it loaded and will keep serving it. The
registry is out of sync.

**The solution — a shared state file**

`score_multi_ort.py` uses a plain JSON file in `/tmp` as a lightweight message
board. Whenever any worker changes the registry it atomically overwrites the
file. Every other worker checks the file's modification time at the start of
each request and reloads if it has changed.

```
  ┌─────────────────────────────────────────────────────┐
  │                     AKS Pod                         │
  │                                                     │
  │  ┌────────────────┐         ┌────────────────┐      │
  │  │   Worker A     │         │   Worker B     │      │
  │  │                │         │                │      │
  │  │ _registry:     │         │ _registry:     │      │
  │  │ {iris, sine}   │         │ {iris, sine}   │      │
  │  └───────┬────────┘         └───────┬────────┘      │
  │          │                          │               │
  │  POST /unload iris          on next request:        │
  │  del _registry["iris"]      getmtime() → changed!   │
  │  os.replace(file) ─────────────────► reload from    │
  │          │                          │  state file   │
  │          ▼                          ▼               │
  │  ┌──────────────────────────────────────────────┐   │
  │  │  /tmp/aml_score_multi_registry.json          │   │
  │  │  { "loaded": ["pytorch_sine"] }              │   │
  │  └──────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────┘
```

Key properties of this approach:

- **Write path** — after every `load` or `unload`, the authoritative worker atomically
  writes `{"loaded": [...]}` to a shared temp file via `os.replace()` (rename is
  atomic on POSIX, so readers always see either the old or the new file, never a
  half-written one).
- **Read path** — at the start of every `run()` call, each worker compares the file's
  `mtime` to its last-seen mtime.  If unchanged, the check costs a single `getmtime()`
  syscall.  If changed, the worker acquires `_registry_lock` and re-syncs.
- **Thread safety** — a `threading.Lock` serialises all registry mutations within a
  single worker process.
- **No external dependency** — only Python stdlib; no Redis, message queue, or
  database required.

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
