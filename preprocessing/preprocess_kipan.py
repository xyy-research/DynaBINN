"""Prepare the KIPAN multi-omics cohort from KICH, KIRC and KIRP Xena files."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

COHORTS = ("KICH", "KIRC", "KIRP")
CNV_AMP_THRESHOLD = 2.5
CNV_DEL_THRESHOLD = 1.5


def map_gene_ids(mapping_file: Path) -> dict[str, str]:
    mapping = pd.read_csv(mapping_file, sep="\t")
    if not {"id", "gene"}.issubset(mapping.columns):
        raise ValueError(f"{mapping_file}: expected 'id' and 'gene' columns")
    mapping = mapping.dropna(subset=["id", "gene"])
    return dict(zip(mapping["id"].astype(str), mapping["gene"].astype(str)))


def tumor_barcodes(values: pd.Index) -> pd.Index:
    return pd.Index(values.astype(str).str[:15])


def read_continuous(paths: list[Path], gene_map: dict[str, str]) -> pd.DataFrame:
    frames = [pd.read_csv(path, sep="\t", index_col=0) for path in paths]
    result = pd.concat(frames, axis=1).T
    result.index = tumor_barcodes(result.index)
    result = result.loc[result.index.str.endswith("-01")]
    result = result.loc[~result.index.duplicated(keep="first")]
    result = result.rename(columns=gene_map)
    result = result.T.groupby(level=0, sort=True).mean().T
    return result


def read_mutations(paths: list[Path]) -> pd.DataFrame:
    events = pd.concat((pd.read_csv(path, sep="\t") for path in paths), ignore_index=True)
    sample_col = "sample" if "sample" in events.columns else "sampleid"
    for column in (sample_col, "gene"):
        if column not in events.columns:
            raise ValueError(f"Somatic-mutation file is missing column: {column}")
    if "effect" in events.columns:
        excluded = {"Silent", "Intron", "3'UTR", "5'UTR", "RNA", "lincRNA", "Synonymous"}
        events = events.loc[~events["effect"].isin(excluded)].copy()
    events[sample_col] = events[sample_col].astype(str).str[:15]
    events = events.loc[events[sample_col].str.endswith("-01")]
    events["mut_value"] = 1
    if events.empty:
        raise ValueError("No tumor mutation records remained after filtering")
    return events.pivot_table(index=sample_col, columns="gene", values="mut_value", aggfunc="max", fill_value=0)


def read_labels(mrna_paths: dict[str, Path]) -> pd.DataFrame:
    labels = []
    for cohort, path in mrna_paths.items():
        header = pd.read_csv(path, sep="\t", index_col=0, nrows=0)
        samples = tumor_barcodes(header.columns)
        tumor = samples[samples.str.endswith("-01")]
        labels.append(pd.DataFrame({"target": cohort}, index=tumor))
    combined = pd.concat(labels)
    combined = combined.loc[~combined.index.duplicated(keep="first")]
    encoder = LabelEncoder()
    combined["label"] = encoder.fit_transform(combined["target"])
    print("Label mapping:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))
    return combined[["label"]]


def run(raw_dir: Path, output_dir: Path, gene_map_file: Path) -> None:
    mrna_paths = {c: raw_dir / f"TCGA-{c}.star_tpm.tsv.gz" for c in COHORTS}
    cnv_paths = [raw_dir / f"TCGA-{c}.gene-level_ascat2.tsv.gz" for c in COHORTS]
    snv_paths = [raw_dir / f"TCGA-{c}.somaticmutation_wxs.tsv.gz" for c in COHORTS]
    required = [gene_map_file, *mrna_paths.values(), *cnv_paths, *snv_paths]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing raw input files:\n" + "\n".join(missing))

    mapping = map_gene_ids(gene_map_file)
    labels = read_labels(mrna_paths)
    mrna = read_continuous(list(mrna_paths.values()), mapping)
    cnv = read_continuous(cnv_paths, mapping)
    snv = read_mutations(snv_paths)
    cnv_amp = (cnv > CNV_AMP_THRESHOLD).astype(np.uint8)
    cnv_del = (cnv < CNV_DEL_THRESHOLD).astype(np.uint8)
    common = labels.index.intersection(mrna.index).intersection(cnv_amp.index).intersection(cnv_del.index).intersection(snv.index)
    if common.empty:
        raise ValueError("No shared tumor sample IDs across the four modalities and labels")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "KIPAN_mRNA_matrix.csv": mrna,
        "KIPAN_CNV_amp_matrix.csv": cnv_amp,
        "KIPAN_CNV_del_matrix.csv": cnv_del,
        "KIPAN_SNV_matrix.csv": snv,
        "KIPAN_Labels.csv": labels,
    }
    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.loc[common].to_csv(path)
        print(f"{filename}: {len(common)} samples x {frame.shape[1]} columns")
    print("KIPAN class counts:", labels.loc[common, "label"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, required=True, help="Directory containing UCSC Xena TSV.GZ files")
    parser.add_argument("--output_dir", type=Path, default=Path("TCGA/KIPAN"))
    parser.add_argument("--gene_map", type=Path, default=None, help="gencode.v36.annotation.gtf.gene.probemap")
    args = parser.parse_args()
    gene_map = args.gene_map or args.raw_dir / "gencode.v36.annotation.gtf.gene.probemap"
    run(args.raw_dir, args.output_dir, gene_map)
