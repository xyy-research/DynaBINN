# DynaBINN

**DynaBINN: Prior-Preserving Topology Adaptation for Hierarchical Multi-Omics Learning**

DynaBINN is a hierarchical multi-omics learning framework that preserves curated Reactome connections while learning a bounded set of additional gene–pathway links. It combines Modality-Adaptive Gating (MAG), Prior-Orthogonal Graph Attention (POGA), and Sparse Cross-Pathway Attention (SCA).

This repository provides the model implementation and scripts for the main classification experiments on BRCA, KIPAN, LGG, and prostate cancer cohorts.

## Repository structure

```text
DynaBINN/
├── train.py                 # Main experiment entry point
├── dynabinn/
│   ├── model.py             # MAG, POGA, SCA, and DynaBINN model
│   ├── data.py              # Multi-omics loading and Reactome hierarchy
│   ├── training.py          # Two-stage learning and five-fold evaluation
│   ├── losses.py            # Focal loss
│   ├── metrics.py           # Evaluation metrics
│   └── utils.py             # Random seed utilities
├── configs/
│   └── datasets.json       # Input file paths for the four cohorts
├── preprocessing/
│   ├── preprocess_kipan.py # KICH/KIRC/KIRP multi-omics preprocessing
│   └── README.md           # Sources and preparation instructions
├── data/
│   ├── ReactomePathways.gmt
│   ├── ReactomePathwaysRelation_new_download.txt
│   └── protein-coding_gene_with_coordinate_minimal.txt
├── tests/
│   └── test_model_smoke.py  # Model forward-pass test
├── environment.yml
└── requirements.txt
```

## Requirements and installation

The reported experiments used Python 3.11 and PyTorch 2.5.1+cu121 on an NVIDIA RTX 4090 D. The remaining Python dependencies are listed in `requirements.txt`.

```bash
conda env create -f environment.yml
conda activate dynabinn
python -m pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with your CUDA environment. The PyTorch installation instructions are available at https://pytorch.org/get-started/locally/.

## Data preparation

The experiments use three TCGA cohorts (BRCA, KIPAN, and LGG) obtained through [UCSC Xena](https://xenabrowser.net/datapages/), the prostate cancer cohort from the [P-NET Zenodo archive](https://zenodo.org/records/5163213), and the Reactome pathway hierarchy. The required reference files are included in the repository's `data/` directory. See [preprocessing/README.md](preprocessing/README.md) for data sources and the KIPAN preprocessing workflow.

Place the preprocessed input matrices under an input root with the following layout:

```text
INPUT_ROOT/
├── data/
│   ├── ReactomePathwaysRelation_new_download.txt
│   ├── ReactomePathways.gmt
│   └── protein-coding_gene_with_coordinate_minimal.txt
└── TCGA/
    ├── BRCA/
    ├── KIPAN/
    ├── LGG/
    └── qlx/       # Prostate cohort (P1000 matrices)
```

The required cohort filenames are defined in `configs/datasets.json`. Each molecular matrix is a CSV file with sample identifiers as rows and genes as columns. The label file must contain a `label` column and corresponding sample identifiers. The four input channels are SNV, CNV amplification, CNV deletion, and mRNA.

The training script reads the prepared cohort matrices directly. For KIPAN, the included preprocessing script combines KICH, KIRC, and KIRP expression, absolute copy-number, and somatic-mutation tables from UCSC Xena:

```bash
python preprocessing/preprocess_kipan.py --raw_dir /path/to/kipan_xena_files --output_dir /path/to/INPUT_ROOT/TCGA/KIPAN
```

Use `--gene_map /path/to/gencode.v36.annotation.gtf.gene.probemap` when the mapping file is stored outside `--raw_dir`. The same five-file input interface is used for BRCA, LGG, and Prostate, with cohort-specific labels. BRCA is classified by PAM50 subtype, while the LGG experiment uses three *histological_type* categories from the UCSC Xena clinical matrix: **Oligodendroglioma (n=192), Astrocytoma (n=193), and Oligoastrocytoma (n=131)**. These labels denote histological categories, not measured IDH mutation or 1p/19q codeletion status. Prostate inputs are derived from the [P-NET dataset on Zenodo](https://zenodo.org/records/5163213). For source details and exact filenames, see [preprocessing/README.md](preprocessing/README.md) and `configs/datasets.json`.

## Running experiments

Run one of the four cohort experiments:

```bash
python train.py --dataset BRCA --input_root /path/to/INPUT_ROOT --save_dir ./outputs
python train.py --dataset KIPAN --input_root /path/to/INPUT_ROOT --save_dir ./outputs
python train.py --dataset LGG --input_root /path/to/INPUT_ROOT --save_dir ./outputs
python train.py --dataset Prostate --input_root /path/to/INPUT_ROOT --save_dir ./outputs
```

Each experiment uses stratified five-fold cross-validation (split seed 10590). Within each fold, 15% of the outer training partition is used for validation. The mRNA channel is log-transformed with `log1p`, and its Min–Max scaler is fitted on the optimization partition and applied to validation and test samples.

Stage 1 learns candidate topology scores for 50 epochs. The 60 highest-scoring non-prior gene-to-first-pathway connections are retained. Stage 2 reinitializes the predictive network and trains on the fixed augmented hierarchy, using early stopping with patience 40 and a maximum of 1500 epochs. SCA retains up to 20 within-level pathway neighbors per query.

## Outputs

Results are saved under `--save_dir/<dataset>/`. Each fold directory contains a model checkpoint, the prior and augmented first-layer masks, held-out sample indices, labels, predicted probabilities, and the fitted mRNA scaler. The experiment also writes a summary of the five-fold classification metrics and cohort metadata.

## Basic checks

```bash
python train.py --help
python -m unittest discover -s tests -v
python -m compileall -q train.py dynabinn tests
```

## Reference data

The Reactome pathway annotations in `data/` are derived from [Reactome](https://reactome.org/download-data/) and provided under [CC0](https://reactome.org/license). The protein-coding gene-coordinate list is included as the gene filtering reference used by the training code.
