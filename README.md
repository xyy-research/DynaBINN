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

The experiments use three TCGA cohorts (BRCA, KIPAN, and LGG) obtained through UCSC Xena, the prostate cancer cohort from the P-NET study, and the Reactome pathway hierarchy.

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

The training script reads the prepared matrices directly. Obtain the cohort datasets and Reactome annotations from their respective sources and prepare them using the preprocessing protocol described in the paper and supplementary material.

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
