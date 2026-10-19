# Data preparation and layout

The data source is AIHub [「위암 병리 이미지 및 판독문 합성데이터」](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=115&dataSetSn=71830&topMenu=100). Obtain the required files under the provider's access conditions and terms; its official page limits data applications to Korean nationals. Confirm that your authorized data cover the real and synthetic IDs in the supplied manifests. Source data are not included here, and this project's MIT code license and CC BY 4.0 weight license do not apply to AIHub data or third-party materials.

The preparation scripts start with selected 1024 × 1024 image patches and their matching JSON annotations. They do not extract patches from whole-slide images or choose a new synthetic subset. The supplied `data/manifests/selected_metadata.csv` contains 750 real and 600 synthetic images per tissue, for 5,400 unique images in total. The 28 training/validation/test manifests in the same directory preserve the fixed selection, patient assignment, and row order.

## Supplied input table

Use `data/manifests/selected_metadata.csv` directly and place the external images and annotations at its paths. It has the following ten columns; the illustrative row below is not an additional study input:

```csv
id,split,source,cancer_type,image_path,label_path,inst_mask_path,class_mask_path,original_split,pid
image_001,train,real,STNT,data/images/image_001.png,data/labels/image_001.json,data/masks/instance/image_001.npy,data/masks/class/image_001.npy,train,pid_0001
```

Image IDs are unique in this table. The four tissue codes are listed below; `source` is `real` or `synthetic`, and `split` is the final `train`, `val`, or `test` assignment. All selected synthetic images belong to `train`. Real rows use patient-group keys in `pid_0000` form, and synthetic rows have an empty `pid`. Each output mask has its own `.npy` path. All paths are relative to the project root. The original patient-identity mapping is not included.

`original_split` records the earlier assignment/storage layout. For 1,011 entries, it differs from the final `split`; this is not a reason to rewrite the CSV or move data to a different split directory. Follow the recorded file paths for storage and the `split` column for experiment membership. Preserve IDs, patient keys, and row order.

Annotation objects are read from `content.file.object`. Each object supplies `label` (or `label_nm`) and a `coordinate` list with at least three `[x, y]` pairs; `{ "x": ..., "y": ... }` points are also supported. Recognized labels are `Stroma`, `Immune`, `Normal`, and `Tumor`. When creating a new patient assignment from annotations, real-image annotations also need a stable patient key at `content.clinical.pid`. The supplied fixed manifests already contain their patient assignments, so this allocation step is not needed to use them.

CSV image and patient IDs retain leading zeros. Annotation patient keys should use a consistent representation: strings are trimmed, and numeric keys such as `7` and `7.0` become `"7"`; the string `"0007"` remains distinct.

## Prepare masks for the fixed manifests

Run from the repository root:

```sh
python -m src.preprocessing.make_masks --meta data/manifests/selected_metadata.csv --output-meta data/meta.csv
```

This reads the fixed table and writes its referenced instance/class masks plus a local metadata copy, preserving all fields and rows. Use the supplied 28 manifests directly afterward; do not rerun patient splitting or regenerate condition lists for the fixed experiment. `--root` selects the project containing the input/output paths. To train with the bundled commands, prepare the files inside this repository. Existing outputs are rejected by default. `--reuse-existing` accepts an existing mask pair only after its arrays and dtypes match masks regenerated from the annotation. Existing output metadata must also have the same columns and rows as the selected input table.

Mask conversion preserves annotation order, rounds each coordinate with Python `round`, and rasterizes polygons with Pillow. Earlier objects keep overlapping pixels. Objects with no unassigned pixels do not receive an instance ID. Coordinates outside the image are clipped during rasterization. No area filter is applied at this stage. Instance masks are `int32`, with 0 for background; class masks are `uint8`, with Stroma=1, Immune=2, Normal=3, Tumor=4. Class masks are retained as annotation outputs, while this study trains only NP/HV branches.

## Preparing a new independent dataset

The patient splitter and condition-list generator remain available for new experiments. Supply a separate selected table with the fields above, and use new output paths:

```sh
python -m src.preprocessing.make_masks --meta data/new_selected_data.csv --output-meta data/new_meta.csv
python -m src.data.make_patient_disjoint_split --meta data/new_meta.csv --output data/new_meta_patient_disjoint.csv
python -m src.data.make_splits --meta data/new_meta_patient_disjoint.csv --output-dir data/new_manifests
```

These optional commands are not required for the supplied study lists and were not used to regenerate them. New inputs must also use separate mask output paths. A new experiment must explicitly select its own manifest root; the distributed study settings continue to use `data/manifests`.

Patient splitting uses the original integer-program formulation and seed 42, with exactly 600/75/75 real images per tissue for train/validation/test. A patient shared by tissue groups receives the same split across groups. It fails if these counts cannot be achieved. A new solver run creates a new assignment; it does not guarantee the historical study assignment. Patient metadata and manifest commands require new output files/directories.

Manifest generation preserves the original ID sorting, seeded shuffling, and nested subset selection. It creates the 28 CSV files for the final five conditions and shared validation/test sets. `real300` is derived from the real rows of `replace50`, preserving their order.

## Training input layout

All input paths must be relative to the project root and resolve inside it.

Each tissue group (`STNT`, `STIN`, `STDI`, `STMX`) requires:

```text
data/manifests/<tissue>/val.csv
data/manifests/<tissue>/test.csv
data/manifests/<tissue>/<condition>/train.csv
```

Required columns are `id,pid,split,source,cancer_type,image_path,inst_mask_path`. The supplied CSVs have all ten columns shown above, including `label_path`, `class_mask_path`, and `original_split`. Image IDs must be unique within each manifest. Use `real` or `synthetic` for `source`, and a stable patient key for each real patient across all manifests. Set `split` to `train`, `val`, or `test` and `cancer_type` to the enclosing tissue code. Optional `label_path` and `class_mask_path` columns must contain valid paths when present. Because they are present in these files, the shared training/evaluation input validator requires all four referenced file types to exist, even though the evaluation calculation itself reads only RGB images and instance masks.

For the study inputs, use 1024 × 1024 images. Pillow reads each image as RGB `uint8`. Each `inst_mask_path` must point to a NumPy `.npy` file containing a two-dimensional nonnegative integer array with the same height and width as its image. Use 0 for background and a distinct positive ID for each nucleus; IDs must fit in `int32`. Binary foreground masks and color-coded PNG masks are not instance-mask inputs.

Example row in `data/manifests/STNT/test.csv`:

```csv
id,pid,split,source,cancer_type,image_path,inst_mask_path
image_001,patient_001,test,real,STNT,data/images/image_001.png,data/masks/image_001.npy
```

| Condition | Real training images | Synthetic training images |
|---|---:|---:|
| base | 600 | 0 |
| real300 | 300 | 0 |
| add | 600 | 600 |
| replace50 | 300 | 300 |
| replace100 | 0 | 600 |

Counts apply to each tissue group. Validation and test each contain 75 real images. The real rows of `base` and `add` must match by image ID, as must the synthetic rows of `replace100` and `add`. Each real or synthetic row in `replace50` must match a row of the same source in `add`. The `real300` rows must equal the real rows of `replace50`, including their order. These comparisons include every CSV field. Real-patient keys must be disjoint across train, validation, and test, including across tissue groups. `src/study.py` checks these constraints.

For training, use the HoVer-Net ImageNet-pretrained preactivation ResNet-50 checkpoint and place it at `pretrained/ImageNet-ResNet50-Preact_pytorch.tar`. The loader requires compatible state entries for the complete `conv0`, `d0`, `d1`, `d2`, and `d3` encoder; generic torchvision ResNet-50 weights do not use this layout. The study evaluation commands require full training checkpoints and matching protocol/run records. Private data, ImageNet weights, and those full training records are not distributed.

The [official HoVer-Net training instructions](https://github.com/vqdang/hover_net#training) provide the pretrained Preact-ResNet50 download link. Place the compatible file at the exact path above. It is a separate external input, outside the validation of this study's 20 final models.

The twenty final models are published on [Zenodo as v1.0.0](https://zenodo.org/records/23465622), DOI [10.5281/zenodo.23465622](https://doi.org/10.5281/zenodo.23465622), under CC BY 4.0; see [the weight license notice](../WEIGHTS_LICENSE.md). Follow the [download and integrity instructions](../README.md#separate-model-weights-and-manifest), then extract `checkpoints.zip` into `checkpoints/` from the repository root. The loader expects `checkpoints/manifest.json` and `checkpoints/<tissue>/<condition>/seed_42.pth`. Use `python -m src.eval.predict` with your own image, or `python -m src.eval.public_evaluate` with the supplied 75-image test manifest and all of its referenced files. These public commands do not require ImageNet initialization or original training records. The Zenodo files do not grant access to the private code repository or external data.

## External inputs and exact reproduction

The fixed selection and split metadata are included. The original images, polygon annotations, masks, and a data download/access agreement are external inputs. Obtain these data and access conditions from the data owner; no replacement selection or patient assignment is needed.

| Input | Availability and purpose |
|---|---|
| `data/manifests/selected_metadata.csv` | Included: 5,400 unique selected image rows, final patient/split assignments, and image/annotation/mask paths. Used directly by mask preparation. Its rows equal the unique rows of the 28 execution manifests. |
| `data/manifests/<tissue>/val.csv`, `test.csv`, `<condition>/train.csv` | Included: 28 fixed CSVs. Training uses all relevant train/validation/test files; public evaluation selects the tissue's `test.csv`. No separate frozen `meta_patient_disjoint.csv` is required. |
| `data/manifests/<tissue>/replace100/train.csv` and the synthetic rows of `add/train.csv` | Included: the identical fixed 600 synthetic selections per tissue. `replace50` uses a fixed 300-row subset. Use these fixed selections directly; initial synthetic-selection automation is not included. |
| `image_path` and `label_path` files | External: selected RGB tiles and polygon JSON. Preserve the exact relative paths stored in the supplied CSVs. |
| `inst_mask_path` and `class_mask_path` files | External or generated with the mask command above: `.npy` instance/class masks at the recorded paths. |
| `pretrained/ImageNet-ResNet50-Preact_pytorch.tar` | External: compatible ImageNet initialization for new training only; this is not one of the 20 final study weights. |

CSV-only checks cover required fields, unique image IDs, per-tissue source/split counts, fixed subset membership, the exact `real300`/`replace50` real-row order, and patient disjointness across tissues and splits. They do not verify image dimensions, annotation-to-mask agreement, actual image content, or correspondence between supplied patient-group keys and the original clinical identities.

`.gitignore` permits only this README and the 29 explicitly listed metadata CSVs under `data/`. The 20 per-condition `config.json` files delivered alongside them are not read by the execution code and remain excluded; runtime settings come from `configs/`. Raw inputs, generated masks, new experiment lists, patient-identity mappings, and internal transfer/verification records remain excluded.
