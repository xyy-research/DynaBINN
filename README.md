# DynaBINN: Prior-Preserving Topology Adaptation for Hierarchical Multi-Omics Learning

Main-experiment research code organized from the provided official training script.
This is a **modular packaging draft**, not a claim of independent reproduction of the paper tables.
The core neural-network classes and fold-training computations were retained; file paths and entry point were reorganized.

## Layout

```text
DynaBINN/
├── train.py                   # short CLI entry point
├── dynabinn/
│   ├── model.py               # MAG (RobustOmicsFusion), POGA, SCA, DynaBINN
│   ├── data.py                # Reactome hierarchy + four-channel loading
│   ├── training.py            # Stage 1 / Top-60 / Stage 2 / 5-fold CV
│   ├── losses.py              # DynamicFocalLoss
│   ├── metrics.py             # classification evaluation
│   └── utils.py               # seed control
├── configs/datasets.json     # BRCA, KIPAN, LGG, Prostate (P1000)
├── requirements.txt          # human-readable unpinned direct dependencies
├── environment.yml           # environment starting point, not exact export
├── scripts/export_environment.sh
├── tests/test_model_smoke.py # synthetic shape check only
└── .gitignore
```

**The data, pretrained weights, and experimental outputs are intentionally excluded.**

## Installation

Use the **working training environment** for best reproduction. From that environment,
export `requirements-locked.txt` and `environment-locked.yml` with
`scripts/export_environment.sh`; inspect files before publication.

For a new environment you may start with:

```bash
conda env create -f environment.yml
conda activate dynabinn
# Install a CPU/CUDA-compatible PyTorch build from https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
```

The packaged environment files are **templates**, not validated exact versions.

## Input files and folder layout

Point `--input_root` to the directory containing **both** `data/` and `TCGA/`.
All cohort filenames are specified in `configs/datasets.json` and can be changed
without editing code. For example:

```text
INPUT_ROOT/
  data/
    ReactomePathwaysRelation_new_download.txt
    ReactomePathways.gmt
    protein-coding_gene_with_coordinate_minimal.txt  # optional in supplied code
  TCGA/
    BRCA/  BRCA_SNV_matrix.csv  BRCA_CNV_amp_matrix.csv  ...
    KIPAN/ KIPAN_SNV_matrix.csv KIPAN_CNV_amp_matrix.csv ...
    LGG/   LGG_SNV_matrix.csv   LGG_CNV_amp_matrix.csv ...
    qlx/   P1000_SNV_matrix.csv P1000_CNV_amp_matrix.csv ...
```

Every modality matrix is a CSV with patient/sample identifiers as the index
and gene names as columns. The label CSV must have a column named `label`.
Users must obtain input data and Reactome sources legally and apply the study's
preprocessing before executing the program; raw datasets are not redistributed.

## Run main classification experiments

```bash
python train.py --dataset BRCA     --input_root /path/to/INPUT_ROOT --save_dir ./outputs
python train.py --dataset KIPAN    --input_root /path/to/INPUT_ROOT --save_dir ./outputs
python train.py --dataset LGG      --input_root /path/to/INPUT_ROOT --save_dir ./outputs
python train.py --dataset Prostate --input_root /path/to/INPUT_ROOT --save_dir ./outputs
```

The `Prostate` key uses the **P1000 files in `TCGA/qlx/`**, precisely as supplied.
The provided implementation selects the classification metrics by the label count.
It uses `log1p` on mRNA values, then fits mRNA MinMax scaling on each outer fold's
**optimization subset**; validation/test reuse the fitted scaler. The stratified
five-fold split seed is **10590**, the validation seed is **555 + zero-based fold index**.
Stage 1 trains candidate-edge logits for 50 epochs and selects 60 non-prior
first-mapping edges. Stage 2 reinitializes the model and trains up to 1500 epochs,
with patience 40. The candidate gates use sigmoid values **without ×20 scaling**.
Each fold saves a checkpoint, prior and augmented masks, indices, labels,
predicted probabilities, and the fold's fitted mRNA scaler.

## Verification

```bash
python train.py --help
python -m unittest discover -s tests -v
python -m compileall -q train.py dynabinn tests
```

These checks do not establish reproduction of the paper's numerical results.
The full experiment requires the original prepared cohort files and environment.

## Scope and caveats

- This repository currently contains **main classification code only**; ablations,
  topology controls, sensitivity plots, and baseline experiments are not included.
- The original code's model class name `UltimatePNet` is retained to avoid changes
  in model semantics and state dictionary key names.
- The code **does not automatically download or preprocess raw TCGA files** into
  the required matrices; the exact source datasets and generation procedure should
  be documented before presenting this as a fully independently reproducible artifact.
- This modularization has not been validated with a full four-cohort training run.
- The input data and label conventions (including the LGG third class) must be
  verified against the final manuscript.

## License and release

Do not add a license without agreement from the code owners and applicable
third-party license checks. Before publishing, remove private URLs, account
credentials, patient-identifying records, and internal filesystem references from
**all commits**, including history.