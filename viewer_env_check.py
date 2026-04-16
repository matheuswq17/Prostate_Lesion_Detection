#!/usr/bin/env python
"""
viewer_env_check.py — Environment validator for the Ararat Viewer lesion inference backend.

Checks all requirements for real MDT_ProstateX inference.
Run this before viewer_infer.py to diagnose missing dependencies.

Usage:
    python viewer_env_check.py

Exit codes:
    0 - All checks passed, inference should work
    1 - One or more critical checks failed
"""

import sys
import os
import importlib

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MDT_DIR = os.path.join(REPO_ROOT, "MDT_ProstateX")
EXP_DIR = os.path.join(MDT_DIR, "experiments", "exp0")


def check(label, fn):
    """Run a check function, print pass/fail, return True if passed."""
    try:
        msg = fn()
        print(f"  [OK]  {label}" + (f": {msg}" if msg else ""))
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


def check_python():
    v = sys.version_info
    if v < (3, 7):
        raise RuntimeError(f"Python >= 3.7 required, found {v.major}.{v.minor}")
    return f"{v.major}.{v.minor}.{v.micro}"


def check_numpy():
    import numpy as np
    return np.__version__


def check_pandas():
    import pandas as pd
    return pd.__version__


def check_torch():
    import torch
    return torch.__version__


def check_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. GPU is required for real inference with MDT_ProstateX. "
            "CPU fallback is not implemented by the MDT model (calls .cuda() unconditionally)."
        )
    return f"CUDA {torch.version.cuda}, device: {torch.cuda.get_device_name(0)}"


def check_batchgenerators():
    import batchgenerators
    return getattr(batchgenerators, "__version__", "installed")


def check_simpleitk():
    import SimpleITK
    return SimpleITK.Version.VersionString()


def check_nms_extension():
    # The NMS CUDA extension is compiled and installed via setup.py
    # It registers as a Python package 'nms' in the MDT custom_extensions
    nms_path = os.path.join(MDT_DIR, "custom_extensions", "nms")
    if not os.path.isdir(nms_path):
        raise RuntimeError(f"NMS source not found at {nms_path}")
    # Try to import the actual built extension
    sys.path.insert(0, nms_path)
    try:
        import nms  # noqa
        return "imported"
    except ImportError as e:
        raise RuntimeError(
            f"NMS CUDA extension not compiled. Run: cd {MDT_DIR} && python setup.py install\n"
            f"  (also requires TORCH_CUDA_ARCH_LIST env var set, e.g. '7.5' for RTX GPUs)\n"
            f"  Import error: {e}"
        )
    finally:
        sys.path.pop(0)


def check_roi_align_extension():
    roi_path = os.path.join(MDT_DIR, "custom_extensions", "roi_align")
    if not os.path.isdir(roi_path):
        raise RuntimeError(f"RoIAlign source not found at {roi_path}")
    sys.path.insert(0, roi_path)
    try:
        import roi_align  # noqa
        return "imported"
    except ImportError as e:
        raise RuntimeError(
            f"RoIAlign CUDA extension not compiled. Run: cd {MDT_DIR} && python setup.py install\n"
            f"  Import error: {e}"
        )
    finally:
        sys.path.pop(0)


def check_mdt_imports():
    """Verify the core MDT modules can be imported."""
    sys.path.insert(0, MDT_DIR)
    sys.path.insert(0, EXP_DIR)
    try:
        import utils.exp_utils  # noqa
        import predictor  # noqa
        import evaluator  # noqa
        return "utils, predictor, evaluator"
    except ImportError as e:
        raise RuntimeError(
            f"MDT core import failed: {e}\n"
            f"  Ensure MDT_ProstateX dependencies are installed and CUDA extensions compiled."
        )
    finally:
        sys.path.pop(0)
        sys.path.pop(0)


def check_exp0_configs():
    """Verify experiment configs can be loaded."""
    sys.path.insert(0, MDT_DIR)
    sys.path.insert(0, EXP_DIR)
    try:
        import utils.exp_utils as utils
        cf_module = utils.import_module("cf", os.path.join(EXP_DIR, "configs.py"))
        cf = cf_module.configs()
        return f"model={cf.model}, dim={cf.dim}, channels={len(cf.channels)}"
    except Exception as e:
        raise RuntimeError(f"Could not load exp0 configs: {e}")
    finally:
        sys.path.pop(0)
        sys.path.pop(0)


def check_checkpoints():
    """Verify at least one fold's checkpoint exists."""
    found = []
    for fold in range(5):
        fold_dir = os.path.join(EXP_DIR, f"fold_{fold}")
        epoch_ranking = os.path.join(fold_dir, "epoch_ranking.npy")
        if os.path.isfile(epoch_ranking):
            import numpy as np
            ranking = np.load(epoch_ranking)
            best_epoch = ranking[0]
            ckpt = os.path.join(fold_dir, f"{best_epoch}_best_checkpoint", "params.pth")
            if os.path.isfile(ckpt):
                found.append(f"fold_{fold}/epoch_{best_epoch}")
    if not found:
        raise RuntimeError(
            f"No valid checkpoints found in {EXP_DIR}/fold_*/\n"
            f"  Each fold_N/ needs epoch_ranking.npy and N_best_checkpoint/params.pth"
        )
    return f"found: {', '.join(found[:3])}{'...' if len(found) > 3 else ''}"


def check_preprocessed_data():
    """Check that the out/ directory has metadata (not the actual images, which are patient-specific)."""
    pp_dir = os.path.join(REPO_ROOT, "out")
    info_df = os.path.join(pp_dir, "info_df.pickle")
    if not os.path.isdir(pp_dir):
        raise RuntimeError(
            f"Preprocessed data directory not found: {pp_dir}\n"
            f"  Run ProstateX preprocessing.ipynb first."
        )
    if not os.path.isfile(info_df):
        raise RuntimeError(
            f"info_df.pickle not found in {pp_dir}\n"
            f"  This is auto-generated on first train run or can be built from meta_info_*.pickle files."
        )
    import pandas as pd
    df = pd.read_pickle(info_df)
    return f"{len(df)} patients in info_df"


def check_img_npy_available():
    """Check if any actual _img.npy files exist (not just metadata)."""
    pp_dir = os.path.join(REPO_ROOT, "out")
    if not os.path.isdir(pp_dir):
        raise RuntimeError(f"out/ directory missing")
    npy_files = [f for f in os.listdir(pp_dir) if f.endswith("_img.npy")]
    if not npy_files:
        raise RuntimeError(
            f"No *_img.npy files found in {pp_dir}\n"
            f"  The preprocessed image arrays are required for inference.\n"
            f"  These are generated by ProstateX preprocessing.ipynb from raw DICOM data.\n"
            f"  For a Viewer-provided patient, pass --pp_dir pointing to the folder that\n"
            f"  contains {{patient_id}}_img.npy (8-channel preprocessed image)."
        )
    return f"{len(npy_files)} image files found (e.g., {npy_files[0]})"


def main():
    print("=" * 60)
    print("Ararat Viewer — Lesion Inference Environment Check")
    print("=" * 60)
    print(f"Repo root: {REPO_ROOT}")
    print(f"MDT dir:   {MDT_DIR}")
    print(f"Exp dir:   {EXP_DIR}")
    print()

    all_passed = True

    print("[1/4] Python & Core Dependencies")
    for label, fn in [
        ("Python version", check_python),
        ("numpy", check_numpy),
        ("pandas", check_pandas),
        ("SimpleITK", check_simpleitk),
    ]:
        if not check(label, fn):
            all_passed = False
    print()

    print("[2/4] PyTorch & CUDA (REQUIRED for inference)")
    for label, fn in [
        ("PyTorch", check_torch),
        ("CUDA available", check_cuda),
        ("batchgenerators", check_batchgenerators),
    ]:
        if not check(label, fn):
            all_passed = False
    print()

    print("[3/4] CUDA Extensions & MDT Imports")
    for label, fn in [
        ("NMS CUDA extension", check_nms_extension),
        ("RoIAlign CUDA extension", check_roi_align_extension),
        ("MDT core imports", check_mdt_imports),
        ("exp0 configs", check_exp0_configs),
    ]:
        if not check(label, fn):
            all_passed = False
    print()

    print("[4/4] Model Weights & Data")
    for label, fn in [
        ("fold checkpoints", check_checkpoints),
        ("preprocessed metadata", check_preprocessed_data),
        ("patient image arrays", check_img_npy_available),
    ]:
        if not check(label, fn):
            all_passed = False
    print()

    print("=" * 60)
    if all_passed:
        print("RESULT: All checks passed. Inference should work.")
        print()
        print("Run inference with:")
        print("  python viewer_infer.py \\")
        print("    --patient_id ProstateX-0001 \\")
        print("    --pp_dir out/ \\")
        print("    --output_dir viewer_output/")
    else:
        print("RESULT: Some checks FAILED. Fix the issues above before running inference.")
        print()
        print("Most common fix for missing PyTorch + CUDA extensions:")
        print("  1. Install PyTorch 1.4+ with CUDA:")
        print("       pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html")
        print("  2. Install batchgenerators:")
        print("       pip install batchgenerators==0.20.1")
        print("  3. Compile custom CUDA extensions:")
        print("       set TORCH_CUDA_ARCH_LIST=7.5  (adjust for your GPU)")
        print(f"       cd {MDT_DIR}")
        print("       python setup.py install")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
