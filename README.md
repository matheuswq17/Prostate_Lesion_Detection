# Prostate mpMRI lesion detection system
This repository contains the code for the paper [Deep Learning system for fully automatic detection, segmentation and Gleason Grade estimation of prostate cancer in multiparametric Magnetic Resonance Images](https://arxiv.org/abs/2103.12650) (currently in preprint), which proposes a fully automatic system that takes prostate multi-parametric magnetic resonance images (mpMRIs) from a prostate cancer (PCa) suspect and, by leveraging the [Retina U-Net detection model](https://arxiv.org/abs/1811.08661), locates PCa lesions, segments them, and predicts their most likely Gleason grade group (GGG). 

This model has been adapated to only use [ProstateX data](https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM-NCI+PROSTATEx+Challenges), achieving an AUC of 0.87 at the challenge online learderboard, hence tying up with the winner of the original [ProstateX challenge](https://doi.org/10.1117/1.jmi.5.4.044501).

Please, cite [the peer-reviewed paper](https://www.nature.com/articles/s41598-022-06730-6) if you use any of the code in this repository:
>Oscar J. Pellicer-Valero, José L. Marenco Jiménez, Victor Gonzalez-Perez, Juan Luis Casanova Ramón-Borja, Isabel Martín García, María Barrios Benito, Paula Pelechano Gómez, José Rubio-Briones, María José Rupérez, José D. Martín-Guerrero, Deep Learning for fully automatic detection, segmentation, and Gleason Grade estimation of prostate cancer in multiparametric Magnetic Resonance Images. Scientific Reports. February, 2022.

This is an example of the output of the model:
![Model output](./media/model_output.png "Model output")
>Output of the model evaluated on three ProstateX test patients. First image from the left shows the GT on the T2 mpMRI sequence; the rest show the output predictions of the model on different sequences (from left to right: T2, b800, ADC, Ktrans). GGG0 (benign) detections are not shown and only the highest-scoring detection is shown for highly overlapped detections (IoU > 0.25). Detections with a confidence below the lesion-wise maximum sensitivity setting (t=0.028) are also ignored.

## Overview
This repository contains three main Jupyter Notebooks:
- [ProstateX preprocessing](ProstateX%20preprocessing.ipynb): Performs the preprocssing of the ProstateX data. All steps, from downloading the data to configuring the Notebook are explained within it. At the end, it creates a `ID_img.npy` file containng the processed image and masks, a `ID_rois.npy` file containing the lesions, and a `meta_info_ID.pickle` containing important meta information for every patient `ID`.
- [Registration example](Registration%20example.ipynb): Notebook for performing general medical image registration, but adapated to register prostate ADC maps to T2 sequences. It takes some unregistered `ID_img.npy` files and produces a SimpleITK transform for each of them as output as `ID.tfm`.
- [Result analysis](./MDT_ProstateX/Result%20analysis.ipynb): It analyzes the results produced by the Retina U-Net model (`processed_pred_boxes_overall_hold_out_list_test.pickle`), allowing to plot the ground truth alongside the predicted detections, obtain metrics and ROC curves at lesion and patient level, and generate a ProstateX challenge submission.

**Note**: the Notebooks should not be run within Jupyter Lab, but rather opened intedependently in Jupyter Notebook, otherwise the `plot_lib` visualizations might not display properly. For more information on the issue, see [this link](https://ipywidgets.readthedocs.io/en/latest/user_install.html#installing-the-jupyterlab-extension).

Additionally, the directory `./MDT_ProstateX` contains a complete fork of the [Medical Detection Toolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit/tree/torch1x), which is employed as the backend for the system and has been modified to adapt it to this task. All modifications to any of the files have been listed in their headers, in compliance with the Apache 2.0 license used by that project.

## Installation
To install, please clone this repository and install required packages. It is recommended to use a package manager such as pip or conda (conda was the only one tested, so use pip at your own risk). If not sure, you can download and install the [latest miniconda release](https://docs.conda.io/en/latest/miniconda.html) before continuing and install `git` from the conda console: `conda install git`

```bash
git clone https://github.com/OscarPellicer/prostate_lesion_detection.git

#You probably want to create an environment first. Using conda:
conda create -n prostate_lesion python=3.7
conda activate prostate_lesion

#Install required libraries. Using conda:
conda install matplotlib numpy ipywidgets ipython scipy pandas==0.25.3 jupyter ipython scikit-learn
conda install SimpleITK==1.2.4 -c simpleitk
conda install pydicom -c conda-forge
#pip install matplotlib numpy ipywidgets ipython scipy simpleitk==1.2.4 pandas==0.25.3 pydicom jupyter ipython scikit-learn
```

You will also need [`plot_lib`](https://github.com/OscarPellicer/plot_lib) for plotting the mpMRIs within the Jupyter Notebooks. To install it, you may simply clone the repository to your home path: 
```bash
git clone https://github.com/OscarPellicer/plot_lib.git
```

Now we navigate to the cloned repository:
```bash
cd prostate_lesion_detection/
```

Then, some zip files will need to be unpacked, in particular:
- `ProstateX_masks.zip`: This contains **automatically generated** ProstateX masks for the whole prostate as well as the central zone, using the model from: [Robust Resolution-Enhanced Prostate Segmentation in Magnetic Resonance and Ultrasound Images through Convolutional Neural Networks](https://doi.org/10.3390/app11020844)
- `ProstateX_transforms.zip`: This contains the transforms for registration of the ProstateX dataset. These were generated using the Notebook [Registration example](Registration%20example.ipynb) from this repository.
- `./MDT_ProstateX/experiments/exp0/test/test_boxes.zip`: This is the output of the model for the test set, which can be analyzed using the Notebook [Result analysis](./MDT_ProstateX/Result%20analysis.ipynb) from this repository.
- `./MDT_ProstateX/experiments/exp0/test/train_boxes.zip`: This is the output of the model for the train set, which can be used to compete in the ProstateX online challenge using the Notebook [Result analysis](./MDT_ProstateX/Result%20analysis.ipynb) from this repository.

All these can be manually unziped, or using the following commands
```bash
unzip \*.zip -d ./
unzip ./MDT_ProstateX/experiments/exp0/test/\*.zip -d ./MDT_ProstateX/experiments/exp0/test/
```

At this point, you should be able to run all the provided Jupyter Notebooks, but you will not be able to use the [Medical Detection Toolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit/tree/torch1x) for training or inference yet, as you still need to install the packages required by it. In summary, you will need to do two things (make sure to run everything from within the `prostate_lesion` conda / pip environment). 

**Note**: If you have any doubts in this step, it might be worth looking at the original documentation of the toolkit ([README](./MDT_ProstateX/README.md)), alhtough some things here differ slightly. For instance, here we use pytorch 1.7, instead of torch 1.4

First, install `pytorch`:
```bash
conda install pytorch==1.7.0 -c pytorch
```

Then, you need to go to https://developer.nvidia.com/cuda-gpus, look at the Compute Capability of your Nvidia GPU, and create an environmental variable with the latest version that your GPU supports. This is required for the correct compilation of some CUDA custom extensions. E.g.: 
```bash
export TORCH_CUDA_ARCH_LIST="6.0"
```

Please note that a CUDA compiler should be already installed in the system for the next step to work, which can be checked by running `which nvcc`, which should return the install path of `nvcc`, or nothing if `nvcc` is not installed.

Finally, go to the `MDT_ProstateX` folder, and run the `setup.py` script to install the rest of required libraries and build the custom extensions:
```bash
cd MDT_ProstateX
python setup.py install
```

## Usage
All the provided Jupyter Notebooks can be run without actually installing the  [Medical Detection Toolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit/tree/torch1x) requirements.

To use the provided model for inference on new data, you will have to first preprocess your images identicaly to how the ProstateX images have been processed using [ProstateX preprocessing](ProstateX%20preprocessing.ipynb). Then, replace the IDs in the `test` key of the `ss_v2` dictionary at the beginning of the file `./MDT_ProstateX/experiments/exp0/data_loader.py` by your own IDs. Finally, run the model in inference mode and aggragate the results:

```bash
python exec.py --mode test --exp_source experiments/exp0 --exp_dir experiments/exp0
python exec.py --mode analysis --exp_source experiments/exp0 --exp_dir experiments/exp0
```

To use the [Medical Detection Toolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit/tree/torch1x) for training, please create a directory within the `MDT_ProstateX/experiments` directory containing a copy of the files: `configs.py`, `custom_transform.py`, and `data_loader.py` from the provided experiment `MDT_ProstateX/experiments/exp0`. These files will have to be modified to fit your needs, or, at the very least, the `ss_v2` dictionary at the beginning of the file `data_loader.py` should be modified to include your own IDs. Then, to run the model and produce the predictions on your test set:

```bash
python exec.py --mode train_test --exp_source experiments/exp1 --exp_dir experiments/exp1
python exec.py --mode analysis --exp_source experiments/exp1 --exp_dir experiments/exp1
```

Either if you use the model for inference or training on your data, you will be able to analyze the results by using the [Result analysis](./MDT_ProstateX/Result%20analysis.ipynb) Notebook. You must remember to change the value of the variable `SUBSET` in this Notebook to point to the new predictions.

## Contact
If you have any problems, please check further instructions in each of the provided Notebooks, create a new Issue, or directly email me at Oscar.Pellicer at uv.es

---

## Ararat Viewer Backend (Real Inference Entrypoint)

This section documents the backend interface between the **Ararat Viewer** and this repository.

### Files

| File | Purpose |
|---|---|
| `inference_server.py` | FastAPI server — wraps `viewer_infer.py` for the Viewer's RemoteInferenceClient |
| `inference_server_config.json` | Runtime config for `inference_server.py` (paths, Python exe) |
| `viewer_infer.py` | Direct inference entrypoint — subprocess-callable, outputs JSON + npy |
| `viewer_env_check.py` | Validates the entire inference environment |
| `viewer_preprocess_stub.py` | Format validator + preprocessing documentation |

### Preprocessed cases — expected structure

The server resolves each patient's data from a single base directory
(`preprocessed_base_dir` in the config).  Each patient gets its own subdirectory:

```
runtime_assets/preprocessed_cases/
  ProstateX-0085/
    ProstateX-0085_img.npy          ← float32 (z, y, x, 8), channels last
    ProstateX-0085_rois.npy         ← float32 (z, y, x, 1)
    meta_info_ProstateX-0085.pickle
    info_df.pickle
  ProstateX-0004/
    ProstateX-0004_img.npy
    ProstateX-0004_rois.npy
    meta_info_ProstateX-0004.pickle
    info_df.pickle
  ...
```

Channels (in order): T2, B500, B800, ADC, Ktrans, Perf_1, Perf_2, Perf_3.

To add a new patient, create its subdirectory with all four required files.
The server will discover it automatically — no config change needed.

### inference_server_config.json

| Key | Description |
|---|---|
| `preprocessed_base_dir` | Path to the folder containing one subdirectory per patient |
| `exp_dir` | Path to `exp0_runtime` (model checkpoints) |
| `output_base_dir` | Where per-job output directories are written |
| `viewer_infer_script` | Absolute path to `viewer_infer.py` |
| `python_exe` | Python executable in the inference conda env |
| `estimated_inference_seconds` | Used to compute pseudo-progress (default 120) |

### Step 1 — Validate the environment

```bash
conda activate prostate_lesion
python viewer_env_check.py
```

This checks Python, PyTorch, CUDA, compiled extensions, checkpoints, and image data.
Fix any `[FAIL]` items before proceeding (see output for specific install commands).

### Step 2 — Populate preprocessed cases

Copy or symlink the preprocessed patient directories into `preprocessed_base_dir`
(the path set in `inference_server_config.json`).

To verify a single patient's files are complete:
```bash
python viewer_preprocess_stub.py runtime_assets/preprocessed_cases ProstateX-0085
```

### Step 3 — Start the server

```bash
conda activate prostate_lesion
python inference_server.py --host 0.0.0.0 --port 8000
```

Verify with:
```bash
curl http://localhost:8000/health
# → {"status":"ok", "preprocessed_base_dir_exists":true,
#    "available_patients_count":2, "available_patients":["ProstateX-0004","ProstateX-0085"], ...}
```

### Step 4 — Submit a job

```bash
curl -X POST http://localhost:8000/api/lesion_inference/run \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"ProstateX-0085","threshold":0.30,"fold":0}'
# → {"status":"submitted","job_id":"<uuid>"}
```

Poll status and download results as usual via `/api/lesion_inference/status/<job_id>`
and `/api/lesion_inference/download/<job_id>/<filename>`.

### Error responses

| Situation | Status | `detail.error` |
|---|---|---|
| Patient directory missing | 404 | `"Patient not found: <id>"` + `available_patients` list |
| Required file(s) missing | 422 | `"Missing required files…"` + `missing_files` list |
| Job not found | 404 | `"Job not found: <id>"` |
| Download before completion | 409 | `"Job not completed yet"` |

### Running viewer_infer.py directly (without the server)

```bash
python viewer_infer.py \
  --patient_id ProstateX-0085 \
  --pp_dir     runtime_assets/preprocessed_cases/ProstateX-0085 \
  --output_dir viewer_output/ProstateX-0085 \
  --fold       0
```

Exit codes: `0` = success, `2` = environment error, `3` = input error, `4` = inference error.

### Output files

| File | Description |
|---|---|
| `lesion_result.json` | Detections: coordinates, scores, Gleason classes |
| `lesion_mask.npy` | Segmentation probability map, shape (H, W, D), float32 |
| `inference_metadata.json` | Run metadata (timing, GPU, shapes) |
| `inference.log` | Full execution log |

### Real limitations (honest)

1. **CUDA required** — the MDT model unconditionally calls `.cuda()`. No CPU fallback.
2. **CUDA extensions must be compiled** — NMS and RoIAlign C++/CUDA extensions need `python setup.py install`.
3. **PyTorch must be installed** — the `prostate_lesion` conda env currently only has pandas + SimpleITK. Install PyTorch 1.7 + CUDA first.
4. **Image must be preprocessed** — raw DICOM → MDT format requires `ProstateX preprocessing.ipynb`. A fully programmatic DICOM pipeline is not yet implemented.
5. **No automatic preprocessing** — new patients must be preprocessed manually and placed in `preprocessed_base_dir/<patient_id>/` before the server can serve them.

### What still needs to be done

- [ ] Install PyTorch + batchgenerators in `prostate_lesion` env and compile CUDA extensions
- [ ] End-to-end test on a real ProstateX patient
- [ ] Implement `preprocess_dicom_to_npy()` in `viewer_preprocess_stub.py` (extracts logic from notebook)
- [ ] Viewer-side integration: load `lesion_mask.npy` as DICOM RT Struct or overlay
- [ ] Multi-fold ensemble (current MVP uses fold 0 only)
