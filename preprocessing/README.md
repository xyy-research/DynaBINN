# Data preparation

DynaBINN uses four molecular channels: non-silent SNV, CNV amplification, CNV deletion, and mRNA expression. The cohort input matrices are indexed by tumor sample ID and contain gene symbols as columns. Labels are stored in a CSV with a `label` column.

## Data sources

- **BRCA, KIPAN, LGG:** [UCSC Xena data pages](https://xenabrowser.net/datapages/), including the GDC Hub datasets for gene expression, gene-level copy number, and somatic mutations.
- **Prostate:** [P-NET data archive on Zenodo (record 5163213)](https://zenodo.org/records/5163213), associated with the P-NET prostate cancer study.
- **Reactome pathway hierarchy:** [Reactome downloads](https://reactome.org/download-data/). Its annotation data are distributed under [CC0](https://reactome.org/license).
- **GENCODE gene annotation mapping:** `gencode.v36.annotation.gtf.gene.probemap`, obtainable through UCSC Xena where applicable.

The exact input filenames are listed in `configs/datasets.json`. The pretrained model weights and cohort input matrices are not required to be checked into the repository.

## KIPAN preprocessing

`preprocess_kipan.py` combines KICH, KIRC, and KIRP, using the following files for each of the three cohorts:

```text
TCGA-<COHORT>.star_tpm.tsv.gz
TCGA-<COHORT>.gene-level_ascat2.tsv.gz
TCGA-<COHORT>.somaticmutation_wxs.tsv.gz
```

Also supply `gencode.v36.annotation.gtf.gene.probemap` (columns `id` and `gene`). The file naming convention and field names follow the original project data preparation.

The preprocessing script:
1. Retains TCGA primary tumor samples with barcode sample type `01`.
2. Converts transcript identifiers to gene symbols, averaging duplicate gene-symbol columns.
3. Encodes non-silent SNV observations as binary values; absent mutation entries are zero for genes present in the mutation matrix.
4. Binarizes ASCAT2 absolute copy number: amplification `>2.5`, deletion `<1.5`.
5. Assigns KICH, KIRC, and KIRP labels based on the source cohort.
6. Intersects sample identifiers across labels and all four modalities, then writes five aligned CSV files.

Run from the repository root:

```bash
python preprocessing/preprocess_kipan.py \
    --raw_dir /path/to/kipan_xena_files \
    --gene_map /path/to/gencode.v36.annotation.gtf.gene.probemap \
    --output_dir /path/to/INPUT_ROOT/TCGA/KIPAN
```

The output files are `KIPAN_SNV_matrix.csv`, `KIPAN_CNV_amp_matrix.csv`, `KIPAN_CNV_del_matrix.csv`, `KIPAN_mRNA_matrix.csv` and `KIPAN_Labels.csv`.

The copy-number thresholds apply to **absolute ASCAT2 copy-number values** and should not be applied to log-ratio data. The resulting mRNA values are subsequently transformed with `log1p` in `dynabinn/data.py`, while fold-specific Min–Max scaling is performed in `dynabinn/training.py`.

## Reactome inputs

The training program expects these files under `INPUT_ROOT/data/`:

```text
ReactomePathways.gmt
ReactomePathwaysRelation_new_download.txt
protein-coding_gene_with_coordinate_minimal.txt
```

The first two are Reactome annotations; the third is the project protein-coding gene-coordinate reference. The code uses this third file when present to intersect eligible genes with protein-coding gene symbols. Retaining the same reference snapshot and gene mapping is important for reproducing the study's exact gene counts.

## BRCA, LGG, and prostate labels

The BRCA and LGG molecular matrices follow the same general alignment, gene mapping, mutation binarization, and absolute-copy-number splitting workflow as KIPAN. The label-generation step differs by cohort:

- **BRCA:** Use PAM50 subtype labels (LumA, LumB, Basal, HER2) from the Xena subtype table (`TCGASubtype.20170308.tsv.gz`, column `Subtype_Selected`), excluding the normal subtype.
- **LGG:** Use `TCGA.LGG.sampleMap_LGG_clinicalMatrix`, column `histological_type`. The experiment uses **Oligodendroglioma (192 samples; encoded 0), Astrocytoma (193; encoded 1), and Oligoastrocytoma (131; encoded 2)**, aligned to the 516 cohort samples in `LGG_Ternary_Labels.csv`. The corresponding mapping is histological; these labels must **not** be interpreted as IDH-mutant/codeleted, IDH-mutant/non-codeleted, or IDH-wildtype classes without separate molecular evidence.
- **Prostate:** Download the P-NET cohort from [Zenodo record 5163213](https://zenodo.org/records/5163213) and prepare Primary/mCRPC labels. The prostate matrices are stored under `TCGA/qlx/` using the `P1000_*.csv` filenames listed in the configuration; this path is a directory convention, not a claim that the cohort is part of TCGA.

The model accepts five cohort-specific CSV files (SNV, CNV amplification, CNV deletion, mRNA, and labels); filenames are defined in `configs/datasets.json`. The published KIPAN script generates KICH/KIRC/KIRP labels and should not be used as the BRCA, LGG, or prostate label generator.
