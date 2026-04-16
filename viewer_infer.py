#!/usr/bin/env python
"""
viewer_infer.py — Real lesion inference backend for the Ararat Viewer.

This is the official contract between the Ararat Viewer and the MDT_ProstateX
prostate lesion detection model. Call this script via subprocess from the Viewer.

──────────────────────────────────────────────────────────
CONTRACT
──────────────────────────────────────────────────────────
INPUT (command-line arguments):
  --patient_id   ProstateX-style ID used in output filenames (e.g. ProstateX-0001)
  --pp_dir       Directory containing {patient_id}_img.npy (8-channel preprocessed image)
                 If {patient_id}_rois.npy is absent, a zeros array is created automatically.
  --output_dir   Directory where results will be written (created if absent)
  --fold         Fold to use for inference (0–4, default: 0)
  --exp_dir      Path to experiment directory (default: MDT_ProstateX/experiments/exp0)
  --debug        Enable verbose logging

OUTPUT (written to --output_dir):
  lesion_result.json          — Machine-readable structured results
  lesion_mask.npy             — Aggregated segmentation probability map, shape (H, W, D)
  inference_metadata.json     — Run metadata (patient id, fold, timing, image shape, …)
  inference.log               — Full text log

──────────────────────────────────────────────────────────
EXIT CODES:
  0  Success — output written to output_dir
  2  Environment error — missing dependency (run viewer_env_check.py)
  3  Input error — missing or malformed input file
  4  Inference error — model crashed during forward pass
──────────────────────────────────────────────────────────
EXAMPLE (from Python subprocess in the Viewer):
  import subprocess, json, numpy as np

  result = subprocess.run([
      "python", "viewer_infer.py",
      "--patient_id", "ProstateX-0001",
      "--pp_dir",     "/path/to/preprocessed",
      "--output_dir", "/tmp/lesion_output",
  ], capture_output=True, text=True)

  if result.returncode == 0:
      import json, numpy as np
      with open("/tmp/lesion_output/lesion_result.json") as f:
          result_data = json.load(f)
      mask = np.load("/tmp/lesion_output/lesion_mask.npy")
  else:
      print("Inference failed:", result.stderr)

──────────────────────────────────────────────────────────
DATA FORMAT REQUIREMENT:
  {patient_id}_img.npy must be a float32 numpy array with the same spatial
  format produced by ProstateX preprocessing.ipynb:
    • 8 channels: T2, B500, B800, ADC, Ktrans, Perf_1, Perf_2, Perf_3
    • Shape: (z, y, x, channels)  [slices-first, channels-last, as from preprocessing.ipynb]
  If you only have partial channels, zero-pad the missing ones to reach 8.

──────────────────────────────────────────────────────────
KNOWN LIMITATIONS:
  1. Requires CUDA — the MDT model calls .cuda() unconditionally.
  2. Custom CUDA extensions (NMS, RoIAlign) must be compiled via:
       cd MDT_ProstateX && python setup.py install
  3. Input image must be preprocessed to the MDT format (see above).
     Raw DICOM → preprocessed .npy requires ProstateX preprocessing pipeline.
  4. PyTorch >= 1.4 required; tested with 1.7 + CUDA 10.x/11.x.
  See viewer_env_check.py for full validation.
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
import traceback
from collections import OrderedDict
from copy import deepcopy

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MDT_DIR = os.path.join(REPO_ROOT, "MDT_ProstateX")
DEFAULT_EXP_DIR = os.path.join(MDT_DIR, "experiments", "exp0")

# These paths are inserted at runtime after argument parsing
_paths_injected = False


def _inject_paths(exp_dir):
    global _paths_injected
    if _paths_injected:
        return
    for p in [MDT_DIR, exp_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)
    _paths_injected = True


# ── Logging ───────────────────────────────────────────────────────────────────

def _build_logger(output_dir, debug=False):
    log_path = os.path.join(output_dir, "inference.log")
    logger = logging.getLogger("viewer_infer")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.addHandler(ch)

    logger.propagate = False
    return logger


# ── Environment guards ────────────────────────────────────────────────────────

def _require(module_name, install_hint):
    try:
        return __import__(module_name)
    except ImportError:
        print(
            f"[ERROR] Required module '{module_name}' not found.\n"
            f"  Install hint: {install_hint}\n"
            f"  Run viewer_env_check.py for a full diagnosis.",
            file=sys.stderr,
        )
        sys.exit(2)


def _check_environment(exp_dir):
    """Fast pre-flight: fail early with clear messages before touching the model."""
    _require("torch", "pip install torch  (>= 1.4, with CUDA)")
    import torch
    if not torch.cuda.is_available():
        print(
            "[ERROR] CUDA not available.\n"
            "  The MDT_ProstateX model requires a CUDA-capable GPU.\n"
            "  CPU-only inference is not supported.",
            file=sys.stderr,
        )
        sys.exit(2)

    _require("batchgenerators", "pip install batchgenerators==0.20.1")
    _require("pandas", "pip install pandas==0.25.3")

    # CUDA extensions
    for ext_name, ext_rel_path in [
        ("nms", os.path.join(MDT_DIR, "custom_extensions", "nms")),
        ("roi_align", os.path.join(MDT_DIR, "custom_extensions", "roi_align")),
    ]:
        if ext_rel_path not in sys.path:
            sys.path.insert(0, ext_rel_path)
        try:
            __import__(ext_name)
        except ImportError:
            print(
                f"[ERROR] CUDA extension '{ext_name}' not compiled.\n"
                f"  Build it with:\n"
                f"    set TORCH_CUDA_ARCH_LIST=7.5  (adjust for your GPU architecture)\n"
                f"    cd {MDT_DIR}\n"
                f"    python setup.py install\n"
                f"  See MDT_ProstateX/Cuda instructions.txt for details.",
                file=sys.stderr,
            )
            sys.exit(2)

    # Checkpoints
    epoch_ranking_path = os.path.join(exp_dir, "fold_0", "epoch_ranking.npy")
    if not os.path.isfile(epoch_ranking_path):
        print(
            f"[ERROR] No epoch_ranking.npy found in {exp_dir}/fold_0/\n"
            f"  Model checkpoints are required for inference.\n"
            f"  Expected path: {epoch_ranking_path}",
            file=sys.stderr,
        )
        sys.exit(2)


# ── Minimal metadata helpers ──────────────────────────────────────────────────

def _ensure_rois_npy(pp_dir, patient_id, img_shape, logger):
    """
    Create a zero-filled rois.npy for inference-only patients (no ground truth).
    Shape matches the img so that PatientBatchIterator does not crash.
    """
    rois_path = os.path.join(pp_dir, f"{patient_id}_rois.npy")
    if os.path.isfile(rois_path):
        logger.info(f"Using existing rois file: {rois_path}")
        return rois_path

    logger.info(
        f"No rois file found for {patient_id}. "
        f"Creating zeros array (inference-only, no ground-truth lesions)."
    )
    # Format: (z, y, x, 1) — matches spatial dims of img, single ROI channel = all background
    z, y, x = img_shape[0], img_shape[1], img_shape[2]
    zeros_rois = np.zeros((z, y, x, 1), dtype=np.float32)
    np.save(rois_path, zeros_rois)
    logger.info(f"Saved zero rois: {rois_path}  shape={zeros_rois.shape}")
    return rois_path


def _ensure_meta_info(pp_dir, patient_id, img_shape, logger):
    """
    Create a minimal meta_info pickle for this patient if not present.
    Uses default values compatible with MDT data loader.
    """
    meta_path = os.path.join(pp_dir, f"meta_info_{patient_id}.pickle")
    if os.path.isfile(meta_path):
        logger.info(f"Using existing meta_info: {meta_path}")
        return meta_path

    logger.info(f"Creating minimal meta_info for {patient_id}")
    meta = {
        "pid": patient_id,
        "class_target": [0],   # unknown / inference-only
        "spacing": (0.5, 0.5, 3.0),  # ProstateX default spacing
        "fg_slices": [],       # no foreground slices known
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    logger.info(f"Saved meta_info: {meta_path}")
    return meta_path


def _ensure_info_df(pp_dir, patient_id, logger):
    """
    Ensure info_df.pickle contains an entry for this patient.
    If the file doesn't exist, creates it with only this patient.
    If it exists, adds the patient if not already present.
    """
    import pandas as pd

    info_df_path = os.path.join(pp_dir, "info_df.pickle")
    meta_path = os.path.join(pp_dir, f"meta_info_{patient_id}.pickle")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    if os.path.isfile(info_df_path):
        df = pd.read_pickle(info_df_path)
        if patient_id not in df["pid"].values:
            new_row = pd.DataFrame([meta])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_pickle(info_df_path)
            logger.info(f"Added {patient_id} to existing info_df ({len(df)} total patients)")
        else:
            logger.info(f"Patient {patient_id} already in info_df")
    else:
        df = pd.DataFrame([meta])
        df.to_pickle(info_df_path)
        logger.info(f"Created info_df with {patient_id} (1 patient)")

    return info_df_path


# ── Inference core ────────────────────────────────────────────────────────────

def _load_net_and_predictor(cf, logger):
    """Load the MDT model + Predictor for test mode."""
    import torch
    import utils.exp_utils as utils
    from predictor import Predictor

    model_module = utils.import_module("model", cf.model_path)
    logger.info(f"Building network: {cf.model} (dim={cf.dim})")
    net = model_module.net(cf, logger).cuda()
    net.eval()

    predictor = Predictor(cf, net, logger, mode="test")
    return net, predictor


def _build_patient_batch_gen(cf, patient_id, img_path, rois_path):
    """Create a single-patient PatientBatchIterator."""
    from data_loader import PatientBatchIterator

    data = OrderedDict()
    data[patient_id] = {
        "data": img_path,
        "seg": rois_path,
        "pid": patient_id,
        "class_target": [0],
        "fg_slices": [],
    }
    batch_gen = {
        "test": PatientBatchIterator(data, cf=cf),
        "n_test": 1,
    }
    return batch_gen


def _extract_viewer_results(results_per_patient, img_shape, cf, logger):
    """
    Convert MDT results_per_patient into Viewer-friendly structures.

    Returns:
        lesion_result: dict  — structured detection results
        seg_mask: np.ndarray — probability map, shape matching img spatial dims
    """
    if not results_per_patient:
        logger.warning("No results returned by predictor.")
        return {"detections": [], "status": "empty"}, np.zeros(img_shape[:3])

    results_dict, pid = results_per_patient[0]
    detections = []

    # boxes is a list over batch elements (batch_size=1 for patient inference)
    for batch_element in results_dict.get("boxes", []):
        for box in batch_element:
            if box.get("box_type") == "det":
                detection = {
                    "box_coords": box["box_coords"].tolist() if hasattr(box["box_coords"], "tolist") else box["box_coords"],
                    "box_score": float(box.get("box_score", 0.0)),
                    "box_label": int(box.get("box_label", 0)),
                    "box_label_name": cf.class_dict.get(int(box.get("box_label", 0)), "unknown"),
                    "box_type": "det",
                }
                detections.append(detection)

    # Sort by score descending
    detections.sort(key=lambda x: x["box_score"], reverse=True)

    # Segmentation probability map: average over batch and ensemble
    seg_preds = results_dict.get("seg_preds", None)
    if seg_preds is not None and seg_preds.size > 0:
        # seg_preds shape: (batch=1, H, W, D) or similar
        seg_mask = np.squeeze(seg_preds).astype(np.float32)
        # Clamp to [0, 1] for probability map
        seg_mask = np.clip(seg_mask, 0.0, 1.0)
    else:
        logger.warning("No segmentation predictions in results. Generating empty mask.")
        seg_mask = np.zeros(img_shape[:3], dtype=np.float32)

    lesion_result = {
        "patient_id": pid,
        "n_detections": len(detections),
        "detections": detections,
        "class_dict": {str(k): v for k, v in cf.class_dict.items()},
        "status": "ok",
    }

    return lesion_result, seg_mask


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ararat Viewer — Real lesion inference backend (MDT_ProstateX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("──")[0],
    )
    parser.add_argument("--patient_id", required=True, help="Patient ID (e.g. ProstateX-0001)")
    parser.add_argument("--pp_dir", required=True, help="Dir containing {patient_id}_img.npy")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--fold", type=int, default=0, choices=range(5), help="Fold (0-4, default 0)")
    parser.add_argument("--exp_dir", default=DEFAULT_EXP_DIR, help="Experiment directory")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = _build_logger(args.output_dir, debug=args.debug)
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("Ararat Viewer — Lesion Inference")
    logger.info(f"  patient_id: {args.patient_id}")
    logger.info(f"  pp_dir:     {args.pp_dir}")
    logger.info(f"  output_dir: {args.output_dir}")
    logger.info(f"  fold:       {args.fold}")
    logger.info(f"  exp_dir:    {args.exp_dir}")
    logger.info("=" * 60)

    # ── Step 1: Environment pre-flight ────────────────────────────────────────
    logger.info("[Step 1] Checking environment...")
    try:
        _check_environment(args.exp_dir)
        logger.info("  Environment OK.")
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Environment check failed: {e}")
        sys.exit(2)

    import torch

    # ── Step 2: Validate input ────────────────────────────────────────────────
    logger.info("[Step 2] Validating input...")
    img_path = os.path.join(args.pp_dir, f"{args.patient_id}_img.npy")
    if not os.path.isfile(img_path):
        logger.error(
            f"Image file not found: {img_path}\n"
            f"  The Viewer must provide a preprocessed 8-channel image in the MDT format.\n"
            f"  See the DATA FORMAT REQUIREMENT section in this script's docstring."
        )
        sys.exit(3)

    try:
        img_data = np.load(img_path, mmap_mode="r")
        img_shape = img_data.shape
        logger.info(f"  Image loaded: {img_path}  shape={img_shape}  dtype={img_data.dtype}")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        sys.exit(3)

    if img_data.ndim != 4:
        logger.error(
            f"Expected 4D array (z, y, x, channels), got shape {img_shape}.\n"
            f"  Ensure the image was preprocessed via ProstateX preprocessing.ipynb."
        )
        sys.exit(3)

    # img format: (z, y, x, channels) — channels is last dim
    expected_channels = 8
    actual_channels = img_shape[3]
    if actual_channels != expected_channels:
        logger.warning(
            f"Image has {actual_channels} channels (last dim), expected {expected_channels}.\n"
            f"  Channels: T2[0], B500[1], B800[2], ADC[3], Ktrans[4], Perf_1[5], Perf_2[6], Perf_3[7].\n"
            f"  Proceeding — the model may underperform with missing channels."
        )

    # ── Step 3: Prepare metadata ──────────────────────────────────────────────
    logger.info("[Step 3] Preparing patient metadata...")
    try:
        _inject_paths(args.exp_dir)
        rois_path = _ensure_rois_npy(args.pp_dir, args.patient_id, img_shape, logger)
        _ensure_meta_info(args.pp_dir, args.patient_id, img_shape, logger)
        _ensure_info_df(args.pp_dir, args.patient_id, logger)
    except Exception as e:
        logger.error(f"Metadata preparation failed: {traceback.format_exc()}")
        sys.exit(3)

    # ── Step 4: Load configs ──────────────────────────────────────────────────
    logger.info("[Step 4] Loading experiment configs...")
    try:
        import utils.exp_utils as utils

        cf_module = utils.import_module("cf", os.path.join(args.exp_dir, "configs.py"))
        cf = cf_module.configs()

        # Override paths for single-patient inference
        cf.exp_dir = args.exp_dir
        cf.fold = args.fold
        cf.fold_dir = os.path.join(args.exp_dir, f"fold_{args.fold}")
        cf.test_dir = os.path.join(args.exp_dir, "test")
        os.makedirs(cf.test_dir, exist_ok=True)

        cf.pp_dir = args.pp_dir
        cf.pp_data_path = args.pp_dir
        cf.pp_test_data_path = args.pp_dir
        cf.input_df_name = "info_df.pickle"
        cf.model_path = os.path.join(args.exp_dir, "model.py")
        cf.max_test_patients = 1

        # Disable data augmentation at test time (test_aug=False is default in configs)
        # Disable plots to avoid matplotlib overhead
        cf.n_workers = 0        # single-threaded for subprocess stability
        cf.debugging = True     # forces SingleThreadedAugmenter

        logger.info(
            f"  Configs: model={cf.model}, dim={cf.dim}, "
            f"fold={cf.fold}, test_n_epochs={cf.test_n_epochs}"
        )
    except Exception as e:
        logger.error(f"Config loading failed: {traceback.format_exc()}")
        sys.exit(2)

    # ── Step 5: Load model + predictor ────────────────────────────────────────
    logger.info("[Step 5] Loading model weights...")
    try:
        torch.cuda.set_device(0)
        with torch.cuda.device(0):
            net, predictor = _load_net_and_predictor(cf, logger)
        logger.info("  Model loaded and moved to CUDA.")
    except Exception as e:
        logger.error(f"Model loading failed: {traceback.format_exc()}")
        sys.exit(4)

    # ── Step 6: Build batch generator ────────────────────────────────────────
    logger.info("[Step 6] Building single-patient batch generator...")
    try:
        batch_gen = _build_patient_batch_gen(cf, args.patient_id, img_path, rois_path)
        logger.info(f"  Batch generator ready for patient: {args.patient_id}")
    except Exception as e:
        logger.error(f"Batch generator failed: {traceback.format_exc()}")
        sys.exit(4)

    # ── Step 7: Run inference ─────────────────────────────────────────────────
    logger.info("[Step 7] Running inference...")
    try:
        with torch.no_grad():
            with torch.cuda.device(0):
                results_per_patient = predictor.predict_test_set(
                    batch_gen, return_results=True, n_test_plots=0
                )
        logger.info(f"  Inference complete. {len(results_per_patient)} result(s) returned.")
    except Exception as e:
        logger.error(f"Inference failed: {traceback.format_exc()}")
        sys.exit(4)

    # ── Step 8: Extract and save results ─────────────────────────────────────
    logger.info("[Step 8] Extracting and saving results...")
    try:
        lesion_result, seg_mask = _extract_viewer_results(results_per_patient, img_shape, cf, logger)

        t_elapsed = time.time() - t_start
        inference_metadata = {
            "patient_id": args.patient_id,
            "fold": args.fold,
            "exp_dir": args.exp_dir,
            "pp_dir": args.pp_dir,
            "output_dir": args.output_dir,
            "img_shape": list(img_shape),
            "seg_mask_shape": list(seg_mask.shape),
            "n_detections": lesion_result["n_detections"],
            "elapsed_seconds": round(t_elapsed, 2),
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "status": "ok",
        }

        # Save files
        result_json_path = os.path.join(args.output_dir, "lesion_result.json")
        with open(result_json_path, "w") as f:
            json.dump(lesion_result, f, indent=2)

        seg_mask_path = os.path.join(args.output_dir, "lesion_mask.npy")
        np.save(seg_mask_path, seg_mask)

        meta_json_path = os.path.join(args.output_dir, "inference_metadata.json")
        with open(meta_json_path, "w") as f:
            json.dump(inference_metadata, f, indent=2)

        logger.info(f"  Saved: {result_json_path}")
        logger.info(f"  Saved: {seg_mask_path}  shape={seg_mask.shape}")
        logger.info(f"  Saved: {meta_json_path}")
        logger.info(f"  Elapsed: {t_elapsed:.1f}s")
        logger.info(f"  Detections: {lesion_result['n_detections']}")
        for i, det in enumerate(lesion_result["detections"][:5]):
            logger.info(
                f"    [{i}] class={det['box_label_name']}  score={det['box_score']:.3f}  "
                f"coords={det['box_coords']}"
            )

    except Exception as e:
        logger.error(f"Result saving failed: {traceback.format_exc()}")
        sys.exit(4)

    logger.info("=" * 60)
    logger.info("Inference COMPLETE. Results in: {}".format(args.output_dir))
    logger.info("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
