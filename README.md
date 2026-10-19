# Gastric nuclei segmentation

HoVer-Net NP/HV instance segmentation for four gastric tissue groups: STNT, STIN, STDI, and STMX.

Version: `v1.0.0`. The [code repository](https://github.com/Jeongyj0/gastric-nuclei-segmentation) is private. Original research code is licensed under MIT; the pretrained weights are separately licensed under CC BY 4.0. See [Licenses](#licenses) for scope and third-party notices.

| Path | Contents |
|---|---|
| `src/preprocessing/`, `src/data/` | Polygon masks, patient splits, condition lists, and training inputs |
| `src/train/`, `src/eval/` | Training, public inference/evaluation, and patient bootstrap statistics |
| `source_core/hovernet/` | Existing model, losses, targets, postprocessing, metrics, and license |
| `configs/` | Study, pilot, and analysis settings |
| `data/manifests/` | The 28 fixed execution CSVs and selected metadata for preprocessing |
| `checkpoints/manifest.json` | Identities, selected updates, sizes, and SHA-256 digests for 20 models |
| `analysis/froc.py` | Instance scoring, IoU matching, and FROC aggregation |
| `tests/` | Executable regression checks used by the launcher/preflight |

The conditions are `base` (Real600), `real300` (Real300), `add` (Real600+Syn600), `replace50` (Real300+Syn300), and `replace100` (Syn600), each trained with seed 42 for each tissue.

## Installation

Run commands from the repository root in a separate virtual environment. Python 3.12.10 was used for CPU execution checks. For example, on Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
```

On Linux/macOS, create the environment with `python3.12 -m venv .venv` and activate it with `source .venv/bin/activate`, then use the same pip commands. These instructions do not constitute a Linux/macOS execution validation.

The observed Windows CPU environment was Python 3.12.10, PyTorch 2.8.0+cpu, NumPy 2.2.6, pandas 2.3.3, Pillow 12.1.0, OpenCV 4.12.0.88, SciPy 1.15.3, scikit-image 0.25.2, and Matplotlib 3.10.8. `requirements-cpu.txt` pins these direct dependencies; it is not a complete transitive environment lock. The existing 20-model checks covered archive integrity, actual CPU loading, and inference on an artificial RGB image. They do not establish pathology performance or equivalence to an original training run.

GPU training requires a CUDA-compatible PyTorch installation; install that build and `requirements.txt` in a separate environment instead of using the CPU requirements. The original training environment, full training, and CUDA execution are not established by these CPU checks.

## Data and preprocessing

The data source is AIHub [「위암 병리 이미지 및 판독문 합성데이터」](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=115&dataSetSn=71830&topMenu=100). Obtain the required RGB tiles and polygon JSON annotations through the provider under its access conditions and terms. The official page limits data applications to Korean nationals; a code or weight license does not grant dataset access. Check that the files supplied under your access agreement cover the real and synthetic image IDs in the manifests; the dataset landing page alone does not establish availability of every study input.

The fixed selection and patient split metadata are included in `data/manifests/`: 28 training/validation/test CSVs and `selected_metadata.csv` covering 5,400 unique images. Source images, annotations, and masks are not bundled. See [data preparation and formats](data/README.md) for the exact fields and paths.

`selected_metadata.csv` contains the fixed 750 real and 600 synthetic 1024 x 1024 image entries per tissue. The same selected synthetic 600 are recorded in each tissue's `replace100/train.csv` and the synthetic rows of `add/train.csv`; a separate selection list is not needed. Use these fixed lists as the selection inputs; initial synthetic-selection automation is not included.

```sh
python -m src.preprocessing.make_masks --meta data/manifests/selected_metadata.csv --output-meta data/meta.csv
```

After placing the images and annotations at their CSV paths, this command creates instance/class masks and a local metadata copy. Use the supplied 28 manifests directly: do not rerun patient assignment or condition-list generation for these fixed inputs. The patient split is 600/75/75 real images per tissue, with patient keys disjoint across all tissues. The scripts for preparing a new independent dataset remain documented in `data/README.md`. Existing outputs are protected from replacement.

Paths retain the original storage layout. Some directory names reflect `original_split` rather than the final `split`; use the CSV `split` field and do not move files or rewrite paths to make those names agree. CSV consistency can be checked without raw data; image dimensions, annotation content, mask validity, and original patient-key correspondence require the external data and were not established by the CSV check.

Training samples four reflected 540 x 540 contexts from each image, augments/crops to 256 x 256, and uses central 164 x 164 NP/HV targets. Inference uses reflected 256 x 256 patches, stitches their 164 x 164 outputs, and crops to the original image size. RGB values enter the model as float values in [0,255]; normalization by 255 is inside the model. This repository does not perform whole-slide extraction, physical-resolution resampling, or stain normalization.

## Separate model weights and manifest

The pretrained weights are published on [Zenodo, version v1.0.0](https://zenodo.org/records/23465622), DOI [10.5281/zenodo.23465622](https://doi.org/10.5281/zenodo.23465622), under CC BY 4.0. The record contains the weight archive, its manifest, and the weight license notice. The code is maintained separately in the private GitHub repository; the Zenodo deposit does not include the code or grant repository access.

Download these three files from the record:

- [checkpoints.zip](https://zenodo.org/records/23465622/files/checkpoints.zip?download=1)
- [manifest.json](https://zenodo.org/records/23465622/files/manifest.json?download=1)
- [WEIGHTS_LICENSE.md](https://zenodo.org/records/23465622/files/WEIGHTS_LICENSE.md?download=1)

Git contains the model manifest and [weight license notice](WEIGHTS_LICENSE.md), but not the 20 weight files or the archive. Models are not distributed through Git LFS.

Save `checkpoints.zip` and the downloaded standalone `manifest.json` in the repository root. Before extraction, verify the archive size and SHA-256 and compare the downloaded manifest with `checkpoints/manifest.json` supplied with the code. Both manifests must have the same bytes and digest. Keep the license notice with the downloaded weights.

The ZIP is 2,503,202,242 bytes with SHA-256 `0ec8c0a93709df749b907c0a88cfa58c1b88c94e919fad065d0be5bc1791bfbd`. The manifest is 9,061 bytes with SHA-256 `91206dd17fa958cb678fb885b5677ba786892e4bc25dab82d9a40a8d62d54cd6`. On PowerShell:

```powershell
Get-Item checkpoints.zip | Select-Object Length
Get-FileHash -Algorithm SHA256 checkpoints.zip
Get-FileHash -Algorithm SHA256 manifest.json
Get-FileHash -Algorithm SHA256 checkpoints/manifest.json
```

Do not proceed if a size or digest differs. Extract from the repository root:

```sh
python -m zipfile -e checkpoints.zip checkpoints
```

The archive's top level contains `manifest.json` and tissue directories, so the resulting paths are:

```text
checkpoints/manifest.json
checkpoints/<tissue>/<condition>/seed_42.pth
```

Do not introduce an extra `checkpoints/checkpoints/` directory. `load_public_model` uses `checkpoints/` under the repository root by default. After comparing the standalone manifest, keep it with your download outside the code distribution; only `checkpoints/manifest.json` is a code-distribution file.

The manifest identifies each tissue, condition, seed, selected update, file path, byte size, SHA-256, tensor count/size, and source-checkpoint digest. Each weight is a complete model state dictionary including BatchNorm statistics/counters, without optimizer, scheduler, or random-generator states. These weights are for inference/evaluation, not resuming training. `load_public_model` verifies file size/hash, tensor metadata, keys, shapes, and dtypes before use.

## Inference

```sh
python -m src.eval.predict --image path/to/image.png --tissue STNT --condition base --output-dir outputs/prediction
```

CPU is the default. Use `--device cuda` for a configured GPU, `--batch-size` for patch batching, and `--checkpoint-dir` to select another directory with the same manifest/weight layout. No training manifests, ImageNet weights, or original run records are required.

```python
from src.eval.predict import load_public_model
model, metadata = load_public_model("STNT", "base", device="cpu")
```

Outputs are `instances.npy` (nonnegative integer instance labels, background 0), `raw_maps.npy` (float32 NP foreground probability, H and V maps), and `prediction.json` (model identity and output summary). Output height/width match the input. Existing prediction files require `--overwrite`. A result containing no nuclei is valid.

## Evaluate released weights and compute statistics/FROC

The supplied `data/manifests/<tissue>/test.csv` selects 75 real images per tissue. Place the referenced image, annotation, instance-mask, and class-mask files inside the project. The supplied CSVs include `label_path` and `class_mask_path`, so the shared input validator checks those files as well, although the numerical evaluation reads only RGB images and instance masks. Required formats are in `data/README.md`. The default manifest root already points to `data/manifests`; `--manifest-root` selects another such directory.

```sh
python -m src.eval.public_evaluate --tissue STNT --condition base --froc --output-dir outputs/public_evaluation
```

For all four tissues and five conditions, use a fresh output directory:

```sh
python -m src.eval.public_evaluate --all --froc --output-dir outputs/public_evaluation_all
python -m src.eval.public_statistics --evaluation-dir outputs/public_evaluation_all --output-dir outputs/public_statistics
```

Both commands accept `--checkpoint-dir`. Public evaluation calls the same inference, tile reassembly, postprocessing, and metric functions as the study evaluator, using the released state dictionaries without ImageNet weights or private training records. Outputs under `<tissue>/<condition>/seed_42/` include `test_summary.json`, `status.json`, and private per-image metrics/input hashes. With `--froc`, existing `analysis/froc.py` functions produce per-instance records and `data/froc_curves/<tissue>_<condition>.csv` within that output directory. Each FROC operating point is checked against the metrics from this evaluation; this is not a comparison to saved historical results.

Public statistics checks model-manifest identities, evaluation source versions, per-image metric hashes, patient pairing, and input image/mask hashes across conditions. It uses the existing patient-cluster bootstrap with `configs/analysis.json` (10,000 repetitions, seed 42) and writes 16 contrasts to `statistics.json`. Original training provenance, initial-state equality, and historical split identity are not reconstructed from released weights. CIs describe evaluation-patient sampling uncertainty conditional on completed seeds; they are exploratory pointwise intervals without multiplicity correction or equivalence claims. The study test set was used in an earlier design review, and overlap with generative-model training patients has not been verified.

## Training and evaluation of new study runs

Use the supplied 28 manifests, provide their referenced data files, and place the HoVer-Net ImageNet preactivation ResNet-50 initialization at `pretrained/ImageNet-ResNet50-Preact_pytorch.tar` before running:

```sh
python start_training.py --plan-only
python start_training.py
python -m src.eval.evaluate --all
python -m src.eval.statistics --config configs/analysis.json
```

The launcher uses the active Python interpreter and performs configuration, runtime, and preflight checks before sequential training. The unchanged configuration selects `best_pq.pth` by validation image-mean PQ with earliest-update ties, using the supplied learning-rate and early-stopping policy. Existing run directories are protected. This research evaluation path requires matching full checkpoints, source hashes, manifests, and protocol/run records generated by training with this code. Use the public commands above for the separately released state dictionaries.

## Checks and distribution scope

```sh
python -m unittest discover -s tests -v
python analysis/froc.py --self-test
```

The test suite uses synthetic fixtures and mocked training/model calls where appropriate; it does not establish full-training success or pathology performance. `tests/` and `src/checks/` are executable dependencies of the training/preflight workflow. Saved validation records are not distributed.

Git includes execution code, configurations, tests, dependency files, essential documentation, the model manifest, the 29 explicitly listed selection/split metadata CSVs, and required copyright notices. Raw images, annotations, masks, private patient-identity mappings, weights, archives, run logs, internal records, credentials, and both `figures/` and `figures_final/` (including plotting code and figure aggregate results) stay outside Git. Runtime settings are in `configs/`; only the fixed CSV inputs are included from `data/manifests/`. Numerical evaluation/FROC code is retained; its generated results are local outputs.

## Licenses

- Original research code and accompanying documentation: [MIT License](LICENSE), `Copyright (c) 2026 Yang Jun Jeong`.
- The twenty pretrained weights identified by the model manifest: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), rights holder Yang Jun Jeong. See [WEIGHTS_LICENSE.md](WEIGHTS_LICENSE.md) for attribution and scope.
- Third-party HoVer-Net code retains its [MIT license and copyright notice](source_core/hovernet/LICENSE), `Copyright (c) 2020 vqdang`. Other third-party notices remain applicable; the root license does not replace them or transfer their ownership.
- AIHub source images, annotations, and other third-party materials remain subject to their providers' terms. Neither the code license nor the weight license grants rights to those external materials.
