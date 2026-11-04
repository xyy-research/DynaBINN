# Data preparation

DynaBINN uses four molecular channels: non-silent SNV, CNV amplification, CNV deletion, and mRNA expression. The cohort input matrices are indexed by tumor sample ID and contain gene symbols as columns. Labels are stored in a CSV with a `label` column.

## Data sources

- **BRCA, KIPAN, LGG:** [UCSC Xena data pages](https://xenabrowser.net/datapages/), including the GDC Hub datasets for gene expression, gene-level copy number, and somatic mutations.
- **Prostate:** [P-NET data archive on Zenodo (record 5163213)](https://zenodo.org/records/5163213), associated with the P-NET prostate cancer study.
- **Reactome pathway hierarchy:** [Reactome downloads](https://reactome.org/download-data/). Its annotation data are distributed under [CC0](https://reactome.org/license).
- **GENCODE gene annotation mapping:** `gencode.v36.annotation.gtf.gene.probemap`, obtainable through UCSC Xena where applicable.

The input filenames for each cohort are listed in `configs/datasets.json`.

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

CNV amplification and deletion are defined using absolute ASCAT2 copy-number thresholds of **> 2.5** and **< 1.5**, respectively. mRNA values are transformed with `log1p` in `dynabinn/data.py` and scaled with a fold-specific Min–Max scaler in `dynabinn/training.py`.

## Reactome inputs

The training program expects these files under `INPUT_ROOT/data/`:

```text
ReactomePathways.gmt
ReactomePathwaysRelation_new_download.txt
protein-coding_gene_with_coordinate_minimal.txt
```

The first two files contain Reactome annotations. The third is the protein-coding gene-coordinate reference used for gene selection.

## BRCA, LGG, and prostate labels

The BRCA and LGG molecular matrices follow the same general alignment, gene mapping, mutation binarization, and absolute-copy-number splitting workflow as KIPAN. The label-generation step differs by cohort:

- **BRCA:** Use PAM50 subtype labels (LumA, LumB, Basal, HER2) from the Xena subtype table (`TCGASubtype.20170308.tsv.gz`, column `Subtype_Selected`), excluding the normal subtype.
- **LGG:** Use `TCGA.LGG.sampleMap_LGG_clinicalMatrix`, column `histological_type`. The experiment uses **Oligodendroglioma (192 samples; encoded 0), Astrocytoma (193; encoded 1), and Oligoastrocytoma (131; encoded 2)**, aligned to the 516 cohort samples in `LGG_Ternary_Labels.csv`.
- **Prostate:** Download the P-NET cohort from [Zenodo record 5163213](https://zenodo.org/records/5163213) and prepare Primary/mCRPC labels. The prepared prostate matrices use the `P1000_*.csv` filenames in `TCGA/qlx/`, as specified in `configs/datasets.json`.

Each cohort provides five CSV files: SNV, CNV amplification, CNV deletion, mRNA, and labels. Their filenames are defined in `configs/datasets.json`.
