"""Run DynaBINN main experiment on BRCA, KIPAN, LGG or Prostate."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import json
from pathlib import Path
import torch
from dynabinn.data import prepare_dataset
from dynabinn.training import run_experiment
from dynabinn.utils import SEED, seed_everything

BASE = Path(__file__).resolve().parent
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["BRCA","KIPAN","LGG","Prostate"], default="BRCA")
    parser.add_argument("--input_root", type=Path, default=BASE)
    parser.add_argument("--save_dir", type=Path, default=BASE/"outputs")
    parser.add_argument("--dataset_config", type=Path, default=BASE/"configs"/"datasets.json")
    args = parser.parse_args()
    configs = json.loads(args.dataset_config.read_text(encoding="utf-8"))
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = prepare_dataset(args.input_root, args.dataset, configs[args.dataset])
    run_experiment(bundle, args.save_dir/args.dataset, device)

if __name__ == "__main__":
    main()
