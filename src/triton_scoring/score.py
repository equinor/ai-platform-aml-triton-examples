"""
Triton proxy scoring script for AML Kubernetes Online Endpoints.

This script is used with the pb23h2 AML-integrated Triton image.
The AML inference server (built into pb23h2) calls init() at startup and run()
for each request.

init()  — starts tritonserver as a subprocess (if not already running)
run()   — proxies KFServing V2 JSON requests to the local Triton HTTP server
"""
import json
import logging
import os
import subprocess
import time

import requests

logger = logging.getLogger(__name__)

TRITON_HTTP_PORT = 8000
MODEL_NAME = "iris_classifier"

_triton_proc = None


def _wait_for_triton(timeout_sec=120):
    url = f"http://localhost:{TRITON_HTTP_PORT}/v2/health/live"
    for i in range(timeout_sec):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                logger.info(f"Triton live after {i + 1}s")
                return True
        except Exception:
            pass
        time.sleep(1)
    logger.error(f"Triton did not become live within {timeout_sec}s")
    return False


def init():
    """Called once at container start by the AML inference server."""
    global _triton_proc

    model_dir = os.environ.get("AZUREML_MODEL_DIR", "")
    logger.info(f"[init] AZUREML_MODEL_DIR = {model_dir!r}")

    # Check whether tritonserver is already running (pb23h2 may auto-start it)
    try:
        r = requests.get(
            f"http://localhost:{TRITON_HTTP_PORT}/v2/health/live", timeout=2
        )
        if r.status_code == 200:
            logger.info("[init] Tritonserver already running — skipping subprocess launch")
            return
    except Exception:
        pass

    # Start tritonserver as a background subprocess
    cmd = [
        "tritonserver",
        f"--model-repository={model_dir}",
        "--allow-http=true",
        "--allow-grpc=false",
        "--log-verbose=1",
    ]
    logger.info(f"[init] Launching: {' '.join(cmd)}")
    _triton_proc = subprocess.Popen(cmd)

    if not _wait_for_triton():
        raise RuntimeError("Tritonserver failed to become live within timeout")

    logger.info("[init] Tritonserver ready")


def run(raw_data):
    """
    Called by AML inference server for each scoring request.

    Accepts KFServing V2 inference JSON and proxies it to the local Triton
    HTTP server. Returns Triton's JSON response as a string.

    Example request body:
    {
      "inputs": [{"name": "float_input", "shape": [1,4],
                  "datatype": "FP32", "data": [[5.1, 3.5, 1.4, 0.2]]}]
    }
    """
    try:
        if isinstance(raw_data, (str, bytes, bytearray)):
            payload = json.loads(raw_data)
        else:
            payload = raw_data

        resp = requests.post(
            f"http://localhost:{TRITON_HTTP_PORT}/v2/models/{MODEL_NAME}/infer",
            json=payload,
            timeout=30,
        )
        return resp.text
    except Exception as exc:
        logger.error(f"[run] Error: {exc}")
        return json.dumps({"error": str(exc)})
