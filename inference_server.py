"""
inference_server.py
===================
FastAPI server that wraps viewer_infer.py for the ARARAT Viewer remote inference pipeline.

API contract (matches RemoteInferenceClient in the viewer):

  GET  /health
       → {"status": "ok", "pp_dir_exists": bool, "exp_dir_exists": bool, "gpu": str}

  POST /api/lesion_inference/run
       Body: {"patient_id": str, "threshold": float, "fold": int}
       → {"status": "submitted", "job_id": str}

  GET  /api/lesion_inference/status/{job_id}
       → {"status": "pending|running|completed|failed",
          "progress_percent": int, "elapsed_seconds": float, "message": str,
          "results": {"output_files": {...}}}   ← only when completed

  GET  /api/lesion_inference/download/{job_id}/{filename}
       → binary stream (lesion_mask.npy / lesion_result.json / inference_metadata.json)

Configuration
-------------
Edit inference_server_config.json next to this file before starting.
Run with:
  python inference_server.py [--host 0.0.0.0] [--port 8000]

Dependencies (install in the inference environment):
  pip install fastapi uvicorn[standard]
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).parent / "inference_server.log", encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger("inference_server")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent.resolve()
_CONFIG_FILE = _SCRIPT_DIR / "inference_server_config.json"


def _load_config() -> dict:
    if _CONFIG_FILE.exists():
        logger.info(f"Config: {_CONFIG_FILE}")
        with open(_CONFIG_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    logger.warning(f"Config file not found — using defaults.  Expected: {_CONFIG_FILE}")
    return {}


_cfg = _load_config()

PP_DIR       = Path(_cfg.get("pp_dir",        _SCRIPT_DIR / "preprocessed"))
EXP_DIR      = Path(_cfg.get("exp_dir",        _SCRIPT_DIR / "MDT_ProstateX" / "experiments" / "exp0"))
OUTPUT_BASE  = Path(_cfg.get("output_base_dir", _SCRIPT_DIR / "inference_job_outputs"))
VIEWER_INFER = Path(_cfg.get("viewer_infer_script", _SCRIPT_DIR / "viewer_infer.py"))
PYTHON_EXE   = _cfg.get("python_exe", sys.executable)
EST_SECONDS  = int(_cfg.get("estimated_inference_seconds", 120))

OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

logger.info(f"PP_DIR={PP_DIR}")
logger.info(f"EXP_DIR={EXP_DIR}")
logger.info(f"OUTPUT_BASE={OUTPUT_BASE}")
logger.info(f"VIEWER_INFER={VIEWER_INFER}")
logger.info(f"PYTHON={PYTHON_EXE}")

# ---------------------------------------------------------------------------
# Job registry (in-memory, keyed by UUID string)
# ---------------------------------------------------------------------------

_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()

# Only these three filenames are served via /download
_ALLOWED_FILES = frozenset(
    ["lesion_mask.npy", "lesion_result.json", "inference_metadata.json"]
)


def _new_job(patient_id: str, threshold: float, fold: int) -> dict:
    jid = str(uuid.uuid4())
    out_dir = str(OUTPUT_BASE / f"job_{jid}")
    os.makedirs(out_dir, exist_ok=True)
    return {
        "job_id":        jid,
        "status":        "pending",
        "patient_id":    patient_id,
        "threshold":     threshold,
        "fold":          fold,
        "created_at":    time.time(),
        "started_at":    None,
        "completed_at":  None,
        "output_dir":    out_dir,
        "message":       "queued",
        "log_tail":      [],
    }


def _patch_job(jid: str, **kw) -> None:
    with _JOBS_LOCK:
        if jid in _JOBS:
            _JOBS[jid].update(kw)


# ---------------------------------------------------------------------------
# Subprocess runner (executed in a daemon thread per job)
# ---------------------------------------------------------------------------

def _run_inference_thread(jid: str) -> None:
    """Spawn viewer_infer.py and track its output."""
    with _JOBS_LOCK:
        job = _JOBS.get(jid)
    if job is None:
        return

    patient_id = job["patient_id"]
    fold       = job["fold"]
    out_dir    = job["output_dir"]

    cmd = [
        PYTHON_EXE, str(VIEWER_INFER),
        "--patient_id", patient_id,
        "--pp_dir",     str(PP_DIR),
        "--output_dir", out_dir,
        "--exp_dir",    str(EXP_DIR),
        "--fold",       str(fold),
    ]

    logger.info(f"[JOB {jid[:8]}] cmd: {' '.join(cmd)}")
    _patch_job(jid, status="running", started_at=time.time(), message="inference running")

    log_lines: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(_SCRIPT_DIR),
        )

        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            log_lines.append(line)
            # Keep only the last 100 lines in memory
            if len(log_lines) > 100:
                log_lines = log_lines[-100:]
            logger.debug(f"[JOB {jid[:8]}] {line}")
            _patch_job(
                jid,
                log_tail=log_lines[-20:],
                message=(line[:120] if line else "running"),
            )

        proc.wait()
        rc = proc.returncode
        logger.info(f"[JOB {jid[:8]}] exit code={rc}")

        if rc == 0:
            _patch_job(
                jid,
                status="completed",
                completed_at=time.time(),
                message="inference completed successfully",
                log_tail=log_lines[-20:],
            )
        else:
            _patch_job(
                jid,
                status="failed",
                completed_at=time.time(),
                message=f"viewer_infer.py exited with code {rc}  "
                        + (log_lines[-1] if log_lines else ""),
                log_tail=log_lines[-20:],
            )

    except Exception as exc:
        logger.exception(f"[JOB {jid[:8]}] unhandled exception")
        _patch_job(
            jid,
            status="failed",
            completed_at=time.time(),
            message=str(exc),
        )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ARARAT Inference Server", version="1.0.0")


class _InferRequest(BaseModel):
    patient_id: str  = Field(...,  description="e.g. ProstateX-0085")
    threshold:  float = Field(0.30, ge=0.0, le=1.0)
    fold:       int   = Field(0,    ge=0,   le=4)


@app.get("/health")
def health():
    """Liveness probe — viewer calls this before every submission."""
    return {
        "status":          "ok",
        "pp_dir_exists":   PP_DIR.exists(),
        "exp_dir_exists":  EXP_DIR.exists(),
        "gpu":             _gpu_summary(),
    }


@app.post("/api/lesion_inference/run")
def submit_job(req: _InferRequest, background_tasks: BackgroundTasks):
    """Accept a job, launch viewer_infer.py in a daemon thread, return job_id."""
    img_path = PP_DIR / f"{req.patient_id}_img.npy"
    if not img_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Preprocessed image not found: {img_path}\n"
                f"Expected  {req.patient_id}_img.npy  in pp_dir={PP_DIR}"
            ),
        )

    job = _new_job(req.patient_id, req.threshold, req.fold)
    jid = job["job_id"]
    with _JOBS_LOCK:
        _JOBS[jid] = job

    t = threading.Thread(
        target=_run_inference_thread,
        args=(jid,),
        daemon=True,
        name=f"infer-{jid[:8]}",
    )
    t.start()

    logger.info(f"[SERVER] Job {jid[:8]} launched  patient={req.patient_id}  fold={req.fold}")
    return {"status": "submitted", "job_id": jid}


@app.get("/api/lesion_inference/status/{job_id}")
def job_status(job_id: str):
    """Return status, pseudo-progress, elapsed time, and last log message."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    status       = job["status"]
    created_at   = job["created_at"]
    started_at   = job.get("started_at")
    completed_at = job.get("completed_at")
    now          = time.time()

    elapsed = (completed_at or now) - (started_at or created_at)

    # Pseudo-progress: ramp 0→95 over EST_SECONDS while running, 100 when done
    if status == "completed":
        pct = 100
    elif status == "failed":
        pct = 0
    elif status == "running" and started_at:
        pct = int(min(95, (elapsed / EST_SECONDS) * 90))
    else:
        pct = 2  # pending

    resp: dict = {
        "status":           status,
        "progress_percent": pct,
        "elapsed_seconds":  round(elapsed, 1),
        "message":          job.get("message", ""),
    }

    if status == "completed":
        resp["results"] = {
            "output_files": {f: f for f in sorted(_ALLOWED_FILES)}
        }

    return resp


@app.get("/api/lesion_inference/download/{job_id}/{filename}")
def download_artifact(job_id: str, filename: str):
    """Stream one artifact file from a completed job to the viewer."""
    if filename not in _ALLOWED_FILES:
        raise HTTPException(status_code=400, detail=f"Unknown file: {filename!r}")

    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job not completed yet (status={job['status']})",
        )

    path = Path(job["output_dir"]) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File missing in job output: {filename}")

    return FileResponse(
        path=str(path),
        media_type="application/octet-stream",
        filename=filename,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu_summary() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "CUDA not available"
    except Exception:
        return "torch unavailable"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="ARARAT Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()

    logger.info(f"Listening on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
