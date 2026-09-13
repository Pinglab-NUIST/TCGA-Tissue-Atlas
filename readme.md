# TCGA Tissue Atlas

Resources for the CENTRAL study: **A comprehensive computational pathology atlas for multi-cancer tumor microenvironment characterization and prognosis prediction** ([paper DOI](https://doi.org/10.1038/s41746-026-03181-5)).

This repository provides the TUZI tissue segmentation model (previously called the General Base Model), precomputed tissue maps and tumor maps, and tissueomic descriptors for downstream research.

[Interactive WSI demo](https://wish.pinglab-nuist.org/wsi-preview/) · [All releases](https://github.com/Pinglab-NUIST/TCGA-Tissue-Atlas/releases) · [License](License)

## Choose a resource

| Resource | What to download | Intended starting point |
| --- | --- | --- |
| [TUZI model](https://github.com/Pinglab-NUIST/TCGA-Tissue-Atlas/releases/tag/tissue-segmentation-v5) | Model checkpoint and three example JPEG ROIs | Run tissue segmentation on image regions |
| [Tissue maps](https://github.com/Pinglab-NUIST/TCGA-Tissue-Atlas/releases/tag/tissuemaps-general-base-model-v5) | One ZIP per TCGA cohort | Analyze precomputed 15-category tissue label maps |
| [Tumor maps](https://github.com/Pinglab-NUIST/TCGA-Tissue-Atlas/releases/tag/tumor-maps) | One ZIP covering eight cohorts | Identify background, non-tumor foreground, and tumor foreground |
| [Tissueomic descriptors](https://github.com/Pinglab-NUIST/TCGA-Tissue-Atlas/releases/tag/tissueomic-descriptors-v1) | `tissueomic_data.pkl` | Start directly from case-level numerical descriptors |

Large files are attached under **Assets** on each release page; they are not included in `git clone` or GitHub's automatically generated **Source code (zip/tar.gz)** downloads. Releases describe different resource types, not successive versions of one bundle: select the specific release above rather than relying on the “Latest” badge.

## Tissueomic descriptors: 3,168 descriptor fields

[Download tissueomic_data.pkl](https://github.com/Pinglab-NUIST/TCGA-Tissue-Atlas/releases/download/tissueomic-descriptors-v1/tissueomic_data.pkl) (115,871,207 bytes; approximately 115.9 MB).

The original pickle is distributed unchanged. It contains **3,437 TCGA case identifiers** and **4,199 stored fields** per case. Of these, **1,031 names begin with `NOFEATURE`**; excluding those fields leaves **3,168 descriptors**. The full file retains both groups so users can inspect the original export. The prefix-based selection below does not perform missing-value filtering, normalization, or study-specific feature selection.

| Pickle key | Structure |
| --- | --- |
| `data` | Dictionary mapping TCGA case IDs (for example, `TCGA-3C-AALI`) to NumPy `float64` arrays of shape `(4199,)` |
| `labelnames` | Ordered list of 4,199 unique field names; array position `j` corresponds to `labelnames[j]` |

The stored names include 15 unsuffixed raw-count fields and 1,046 fields for each of `[inside-tumor]`, `[outside-tumor]`, `[invasive-margin-80]`, and `[invasive-margin-500]`. Preserve these names when selecting descriptors. See the paper and supplementary materials for biological definitions, formulas, and region construction; the file itself does not encode a full data dictionary or cohort labels.

**What does `NOFEATURE` mean here?** It is a field-name prefix distinguishing the 1,031 fields excluded from the 3,168-descriptor selection. Their names include auxiliary/raw measurements such as `NOFEATURE_Raw_Count_TYPE_1`, tissue area/count fields, and intersection/ring area/count fields. They are retained in the original export for inspection; the prefix does **not** mean a missing value, an invalid case, or an all-zero column. The field names support this distinction, but the pickle does not provide a separate semantic definition for every auxiliary field. Use the ordered names and the prefix filter below rather than dropping columns by position.

### Load and select descriptors

Requires Python and NumPy; loading and this selection were checked with Python 3.8.12 and NumPy 1.23.5. Only load pickle files from a trusted source, because unpickling can execute code.

```python
import pickle
import numpy as np

with open("tissueomic_data.pkl", "rb") as f:
    resource = pickle.load(f)

case_ids = list(resource["data"])
field_names = np.asarray(resource["labelnames"])
X_all = np.stack([resource["data"][case_id] for case_id in case_ids])
assert X_all.shape == (3437, 4199)
assert len(field_names) == X_all.shape[1]

keep = np.array([not name.startswith("NOFEATURE") for name in field_names])
X = X_all[:, keep]
descriptor_names = field_names[keep]
assert X.shape == (3437, 3168)
print("Cases, descriptors:", X.shape)
print("Missing descriptor entries:", np.isnan(X).sum())

# Optional labeled table (requires pandas):
# import pandas as pd
# descriptors = pd.DataFrame(X, index=case_ids, columns=descriptor_names)
# descriptors.index.name = "case_id"
```

The complete 4,199-field matrix contains **3,558,016 NaN entries** and no infinite values. Missingness is retained, not silently replaced with zeros. Select an appropriate missing-data policy for your analysis; fit any imputation, scaling, and feature selection on training data only. Match external metadata by case identifier, not by row order. These are case-level IDs, whereas map filenames can identify individual slides; do not assume a one-to-one mapping or an unverified slide-aggregation rule.

SHA-256 of the original file:

```text
3d6cedcacd9de375ed9f8bfad27821695523da5359cfed047ef0061c4ca86e92
```

## TUZI tissue segmentation model

[Model release](https://github.com/Pinglab-NUIST/TCGA-Tissue-Atlas/releases/tag/tissue-segmentation-v5) · [Minimal example](TUmor-Zonal-Informatics/General-base-model-v5/minimal-example.py)

The release contains `general_base_model_0.5_v5@108.pth` (approximately 99.7 MB) and three JPEG ROIs from BRCA, COAD, and PAAD. The example defines a VGG16-based U-Net with coordinate attention and 15 output channels. It uses RGB tensors with ImageNet normalization (mean `[0.485, 0.456, 0.406]`, standard deviation `[0.229, 0.224, 0.225]`) and obtains class indices by taking `argmax` across channels.

1. Download the checkpoint and **all three JPEG assets** from the model release.
2. Place them beside `minimal-example.py` in `TUmor-Zonal-Informatics/General-base-model-v5/`.
3. Install a compatible PyTorch/torchvision pair, NumPy, Pillow, and Matplotlib in your Python environment.
4. Run from that directory:

```bash
python minimal-example.py
```

The current script first runs a random 512 × 512 tensor, then displays predictions for the three ROIs. It is a patch-level demonstration, not a complete whole-slide tiling/stitching pipeline. For analysis workflows, use `model.eval()` and `torch.no_grad()` when running inference. Checkpoint loading may require `map_location="cpu"` on CPU-only machines. The existing script is provided as-is; a complete dependency lockfile and end-to-end environment validation are not supplied here.

Class channel indices and the visualization palette are not a substitute for a biological label dictionary. Consult the paper's supplementary materials for category definitions; do not infer tissue identity solely from display colors. The example references the [CoordAttention implementation](https://github.com/Andrew-Qibin/CoordAttention); third-party components retain their applicable terms.

## Precomputed tissue maps

[Tissue-map release](https://github.com/Pinglab-NUIST/TCGA-Tissue-Atlas/releases/tag/tissuemaps-general-base-model-v5)

The release covers eight TCGA cohorts. Its release notes specify 15-category label maps generated at **0.5 µm/pixel**, originally stored as TIFF, then downsampled by **16×** and distributed as palette PNGs. The distributed tissue maps therefore correspond to **8.0 µm/pixel**. They are reduced-resolution label maps, not the original whole-slide images or full-resolution TIFF predictions.

| Cohort | ZIP download | Approximate size (decimal MB) |
| --- | --- | ---: |
| BRCA | [BRCA.zip](https://github.com/Pinglab-NUIST/TCGA-Tissue-Atlas/releases/download/tissuemaps-general-base-model-v5/BRCA.zip) | 989.6 |
| COAD | [COAD.zip](https://github.com/Pinglab-NUIST/TCGA-Tissue-Atlas/releases/download/tissuemaps-general-base-model-v5/COAD.zip) | 348.3 |
| LUAD | [LUAD.zip](https://github.com/Pinglab-NUIST/TCGA-Tissue-Atlas/releases/download/tissuemaps-general-base-model-v5/LUAD.zip) | 656.3 |
| LUSC | [LUSC.zip](https://github.com/Pinglab-NUIST/TCGA-Tissue-Atlas/releases/download/tissuemaps-general-base-model-v5/LUSC.zip) | 590.4 |
| OV | [OV.zip](https://github.com/Pinglab-NUIST/TCGA-Tissue-Atlas/releases/download/tissuemaps-general-base-model-v5/OV.zip) | 124.0 |
| PAAD | [PAAD.zip](https://github.com/Pinglab-NUIST/TCGA-Tissue-Atlas/releases/download/tissuemaps-general-base-model-v5/PAAD.zip) | 209.2 |
| STAD | [STAD.zip](https://github.com/Pinglab-NUIST/TCGA-Tissue-Atlas/releases/download/tissuemaps-general-base-model-v5/STAD.zip) | 420.0 |
| UCEC | [UCEC.zip](https://github.com/Pinglab-NUIST/TCGA-Tissue-Atlas/releases/download/tissuemaps-general-base-model-v5/UCEC.zip) | 839.7 |

Read the stored indices without converting a palette image to RGB:

```python
from PIL import Image
import numpy as np

with Image.open("path/to/extracted_map.png") as image:
    print("Mode:", image.mode, "Size:", image.size)
    labels = np.array(image)  # Preserve stored indices; do not convert to RGB.
assert labels.ndim == 2, "Expected a single-channel or palette label map"
print("Stored values:", np.unique(labels))
```

When resizing categorical maps, use nearest-neighbor interpolation. Verify slide identity, map dimensions, scale, and origin before aligning maps with a WSI or with each other.

## Precomputed tumor maps

[Tumor-map release](https://github.com/Pinglab-NUIST/TCGA-Tissue-Atlas/releases/tag/tumor-maps) · [Download Tumor-Maps-for-8-cohorts.zip](https://github.com/Pinglab-NUIST/TCGA-Tissue-Atlas/releases/download/tumor-maps/Tumor-Maps-for-8-cohorts.zip) (approximately 256.3 MB).

One ZIP covers BRCA, COAD, LUAD, LUSC, OV, PAAD, STAD, and UCEC. The existing release is marked as a prerelease on GitHub. According to its release notes, PNG values mean:

| Value | Meaning |
| --- | --- |
| `0` | Pre-filtered background |
| `1` | Non-tumor region in the foreground |
| `2` | Tumor region in the foreground |

For **LUAD and LUSC**, the notes explicitly document downsampling from **0.5 to 8.0 µm/pixel**. For **BRCA, COAD, OV, PAAD, STAD, and UCEC**, they document direct TIFF-to-PNG conversion but do not state a numerical pixel size. Do not assume all tumor maps have the same resolution as the tissue maps. Confirm the relevant spatial scale from the study materials before area measurements or image registration. The PNG loading example above also applies to these label maps.

## Supporting tables

The repository's [CENTRAL/data](CENTRAL/data/) directory also includes:

- [Slideinfo summary and QC.xlsx](CENTRAL/data/Slideinfo%20summary%20and%20QC.xlsx)
- [spearman_results_160sig.xlsx](CENTRAL/data/spearman_results_160sig.xlsx)
- [spearman_results_33sig.xlsx](CENTRAL/data/spearman_results_33sig.xlsx)

Inspect the workbook headers and the supplementary materials for their field definitions and analysis context. These tables are separate from the descriptor pickle; verify join keys and inclusion criteria before combining them.

## Citation and permitted use

Please cite **A comprehensive computational pathology atlas for multi-cancer tumor microenvironment characterization and prognosis prediction**, DOI [10.1038/s41746-026-03181-5](https://doi.org/10.1038/s41746-026-03181-5), when your research materially uses these resources. No co-authorship is required solely for using the resources.

Project-owned resources in this repository and the descriptor asset released under `tissueomic-descriptors-v1` are provided under the [TCGA-Tissue-Atlas Noncommercial Research License v1.0](License). Free noncommercial research includes model training, fine-tuning, distillation, and development of alternative research models. Commercial use, including commercial product development and commercial use of resulting technical artifacts, requires prior written authorization. Third-party assets and original TCGA/GDC data retain their applicable terms; this notice does not revoke any previously validly granted rights. Consult the specific release notices for external assets.

For licensing requests, contact **ping@nuist.edu.cn**. For questions about files, loading, or interpretation, please [open an issue](https://github.com/Pinglab-NUIST/TCGA-Tissue-Atlas/issues) and include the release tag, asset filename, and relevant error message or case/slide identifier.
