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
  Response → KFServing V2 JSON  {"outputs": [{"name": "label", ...}, {"name": "probabilities", ...}]}
  This is identical to what the Triton ONNX Runtime backend would return.
"""
import glob
import json
import logging
import os

import joblib
import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME  = "iris_classifier"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

_model = None


def init():
    """Called once at container start by the AML inference server."""
    global _model

    model_dir = os.environ.get("AZUREML_MODEL_DIR", "")
    logger.info(f"[init] AZUREML_MODEL_DIR = {model_dir!r}")

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


def run(raw_data):
    """
    Called by the AML inference server for each scoring request.

    Accepts KFServing V2 inference JSON and returns a V2-compatible response.

    Example request body:
    {
      "inputs": [{"name": "float_input", "shape": [1, 4],
                  "datatype": "FP32", "data": [[5.1, 3.5, 1.4, 0.2]]}]
    }
    """
    try:
        if isinstance(raw_data, (str, bytes, bytearray)):
            payload = json.loads(raw_data)
        else:
            payload = raw_data

        # Extract float_input tensor from the V2 request
        X = None
        for inp in payload.get("inputs", []):
            if inp["name"] == "float_input":
                X = np.array(inp["data"], dtype=np.float32).reshape(inp["shape"])
                break

        if X is None:
            raise ValueError("Input 'float_input' not found in request")

        # Sklearn inference (identical result to Triton ONNX Runtime backend)
        labels = _model.predict(X).astype(np.int64).tolist()
        probas = _model.predict_proba(X).astype(np.float32).tolist()

        batch_size = len(labels)
        n_classes  = len(probas[0])

        # Build KFServing V2 response — same shape/dtype as Triton would return
        response = {
            "model_name":    MODEL_NAME,
            "model_version": "1",
            "outputs": [
                {
                    "name":     "label",
                    "shape":    [batch_size],
                    "datatype": "INT64",
                    "data":     labels,
                },
                {
                    "name":     "probabilities",
                    "shape":    [batch_size, n_classes],
                    "datatype": "FP32",
                    "data":     [p for row in probas for p in row],
                },
            ],
        }
        return json.dumps(response)

    except Exception as exc:
        logger.error(f"[run] Error: {exc}", exc_info=True)
        return json.dumps({"error": str(exc)})
