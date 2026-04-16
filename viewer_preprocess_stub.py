#!/usr/bin/env python
"""
viewer_preprocess_stub.py — Preprocessing bridge for the Ararat Viewer.

STATUS: STUB — this module documents what preprocessing is required and
provides helpers for validation and format conversion.
Full programmatic preprocessing from DICOM is NOT yet implemented here.
It still depends on ProstateX preprocessing.ipynb.

──────────────────────────────────────────────────────────
WHY THIS FILE EXISTS
──────────────────────────────────────────────────────────
The MDT_ProstateX model requires images in a specific 8-channel format
produced by ProstateX preprocessing.ipynb. The Viewer must either:

  Option A (MVP): Provide pre-processed .npy files (already done for ProstateX patients)
  Option B (future): Preprocess new patients from DICOM using the pipeline here

This stub covers Option A validation and lays the groundwork for Option B.

──────────────────────────────────────────────────────────
REQUIRED IMAGE FORMAT (from ProstateX preprocessing.ipynb)
──────────────────────────────────────────────────────────
File:  {patient_id}_img.npy
Shape: (x, y, z, 8)  — float32, channels last
Channels:
  [0] T2_TSE_TRA      — T2-weighted transverse
  [1] B500            — DWI b=500
  [2] B800            — DWI b=800 (or B1400)
  [3] ADC             — Apparent Diffusion Coefficient
  [4] Ktrans          — DCE-MRI Ktrans perfusion map
  [5] DCE_Perf_1      — DCE perfusion frame 1
  [6] DCE_Perf_2      — DCE perfusion frame 2
  [7] DCE_Perf_3      — DCE perfusion frame 3

Spacing: isotropic in-plane 0.5mm, slice 3.0mm (resampled)
Registration: ADC, DWI, DCE all registered to T2 space

File:  {patient_id}_rois.npy   (optional for inference)
Shape: same as _img.npy  (zeros = no ground truth)

File:  meta_info_{patient_id}.pickle
Format: {'pid': str, 'class_target': list, 'spacing': tuple, 'fg_slices': list}

──────────────────────────────────────────────────────────
PREPROCESSING PIPELINE SUMMARY (from ProstateX preprocessing.ipynb)
──────────────────────────────────────────────────────────
1. Load DICOM series via SimpleITK + pydicom
2. Identify sequences by series description:
     - t2_tse_tra → T2
     - ep2d_diff (b=500, b=800) → DWI
     - ep2d_diff_ADC → ADC
     - dyn (DCE series) → extract Ktrans + Perf frames
3. Register all sequences to T2 space:
     - Use reg_lib.py (SimpleITK-based rigid/deformable registration)
     - Resample to target spacing (0.5, 0.5, 3.0) mm
4. Normalize each channel independently (min-max or z-score)
5. Stack channels into (x, y, z, 8) array
6. Save as {patient_id}_img.npy
7. Compute ROI masks from ProstateX annotations → {patient_id}_rois.npy
8. Save metadata → meta_info_{patient_id}.pickle

──────────────────────────────────────────────────────────
WHAT'S NEEDED TO COMPLETE OPTION B (fully programmatic)
──────────────────────────────────────────────────────────
1. Extract preprocessing logic from ProstateX preprocessing.ipynb into functions
2. Handle arbitrary DICOM input (not just ProstateX-format naming conventions)
3. Map Viewer's series types to MDT channel indices
4. Handle missing sequences gracefully (zero-fill missing channels)
5. Expose a function: preprocess_dicom_to_npy(dicom_root, patient_id, output_dir)
"""

import os
import pickle
import numpy as np


def validate_img_npy(img_path: str) -> dict:
    """
    Validate that a preprocessed image file meets MDT requirements.

    Returns a dict with:
      - valid: bool
      - shape: tuple
      - dtype: str
      - issues: list of strings describing problems
    """
    issues = []
    result = {"valid": False, "shape": None, "dtype": None, "issues": issues}

    if not os.path.isfile(img_path):
        issues.append(f"File not found: {img_path}")
        return result

    try:
        arr = np.load(img_path, mmap_mode="r")
    except Exception as e:
        issues.append(f"Cannot load as numpy array: {e}")
        return result

    result["shape"] = arr.shape
    result["dtype"] = str(arr.dtype)

    if arr.ndim != 4:
        issues.append(
            f"Expected 4D array (z, y, x, channels), got {arr.ndim}D shape {arr.shape}"
        )
    else:
        z, y, x, c = arr.shape   # actual format: slices-first, channels-last
        if c != 8:
            issues.append(
                f"Expected 8 channels (last dim), got {c}. "
                f"Channels: T2, B500, B800, ADC, Ktrans, Perf_1, Perf_2, Perf_3"
            )
        if z < 5:
            issues.append(f"Suspicious number of slices: z={z}. Expected >= 5 for prostate MRI.")
        if x < 64 or y < 64:
            issues.append(f"Suspicious spatial dimensions: y={y}, x={x}. Expected >= 64.")
        if arr.dtype not in [np.float32, np.float64]:
            issues.append(
                f"Dtype {arr.dtype} is unusual. Expected float32. "
                f"This may or may not be a problem depending on preprocessing."
            )

    result["valid"] = len(issues) == 0
    return result


def validate_meta_info(meta_path: str) -> dict:
    """Validate a meta_info pickle file."""
    issues = []
    result = {"valid": False, "meta": None, "issues": issues}

    if not os.path.isfile(meta_path):
        issues.append(f"File not found: {meta_path}")
        return result

    try:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
    except Exception as e:
        issues.append(f"Cannot unpickle: {e}")
        return result

    required_keys = {"pid", "class_target", "spacing", "fg_slices"}
    missing = required_keys - set(meta.keys())
    if missing:
        issues.append(f"Missing keys: {missing}")

    result["meta"] = meta
    result["valid"] = len(issues) == 0
    return result


def summarize_patient_data(pp_dir: str, patient_id: str) -> None:
    """
    Print a human-readable summary of a patient's preprocessed data.
    Useful for debugging before running viewer_infer.py.
    """
    print(f"\nPatient data summary: {patient_id}")
    print(f"  Directory: {pp_dir}")

    img_path = os.path.join(pp_dir, f"{patient_id}_img.npy")
    v = validate_img_npy(img_path)
    if v["valid"]:
        print(f"  [OK]  _img.npy  shape={v['shape']}  dtype={v['dtype']}")
    else:
        print(f"  [FAIL] _img.npy: {'; '.join(v['issues'])}")

    rois_path = os.path.join(pp_dir, f"{patient_id}_rois.npy")
    if os.path.isfile(rois_path):
        rois = np.load(rois_path, mmap_mode="r")
        n_lesions = int(np.max(rois)) if rois.max() > 0 else 0
        print(f"  [OK]  _rois.npy  shape={rois.shape}  n_lesions={n_lesions}")
    else:
        print(f"  [--]  _rois.npy  (not found — will be created as zeros for inference)")

    meta_path = os.path.join(pp_dir, f"meta_info_{patient_id}.pickle")
    v = validate_meta_info(meta_path)
    if v["valid"]:
        m = v["meta"]
        print(
            f"  [OK]  meta_info  class_target={m['class_target']}  "
            f"spacing={m['spacing']}  fg_slices={len(m['fg_slices'])}"
        )
    else:
        print(f"  [--]  meta_info  (not found — will be created with defaults)")

    info_df = os.path.join(pp_dir, "info_df.pickle")
    if os.path.isfile(info_df):
        import pandas as pd
        df = pd.read_pickle(info_df)
        in_df = patient_id in df["pid"].values
        print(f"  [{'OK' if in_df else '--'}]  info_df.pickle  ({len(df)} total, patient {'present' if in_df else 'absent — will be added'})")
    else:
        print(f"  [--]  info_df.pickle  (will be created)")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# FUTURE: programmatic DICOM → MDT preprocessing
# Uncomment and complete when ready to support new patients from raw DICOM
# ──────────────────────────────────────────────────────────────────────────────

# def preprocess_dicom_to_npy(
#     dicom_root: str,
#     patient_id: str,
#     output_dir: str,
#     t2_series_desc: str = "t2_tse_tra",
#     adc_series_desc: str = "ep2d_diff_ADC",
# ) -> str:
#     """
#     Convert patient DICOM series to MDT-compatible _img.npy.
#     NOT YET IMPLEMENTED — requires extraction from ProstateX preprocessing.ipynb.
#
#     Args:
#         dicom_root:      Root directory of the patient's DICOM files
#         patient_id:      ID to use for output filenames
#         output_dir:      Where to write {patient_id}_img.npy
#         t2_series_desc:  DICOM SeriesDescription for the T2 sequence
#         adc_series_desc: DICOM SeriesDescription for the ADC sequence
#
#     Returns:
#         Path to the written {patient_id}_img.npy
#     """
#     raise NotImplementedError(
#         "Full DICOM preprocessing not yet implemented programmatically.\n"
#         "Use ProstateX preprocessing.ipynb for now, then point --pp_dir to the output."
#     )


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        summarize_patient_data(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python viewer_preprocess_stub.py <pp_dir> <patient_id>")
        print("Example: python viewer_preprocess_stub.py out/ ProstateX-0001")
