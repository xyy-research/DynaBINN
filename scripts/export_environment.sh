#!/usr/bin/env bash
set -euo pipefail
# Activate the environment used for the reported experiments before running.
python -m pip freeze > requirements-locked.txt
if command -v conda >/dev/null 2>&1; then
  conda env export --no-builds | sed '/^prefix: /d' > environment-locked.yml
  conda env export --from-history | sed '/^prefix: /d' > environment-history.yml
fi
python - <<'PY_CHECK'
import torch, numpy, pandas, networkx, sklearn, joblib
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("numpy:", numpy.__version__, "pandas:", pandas.__version__)
print("networkx:", networkx.__version__, "scikit-learn:", sklearn.__version__)
print("joblib:", joblib.__version__)
PY_CHECK
echo "Review exported files for private URLs, paths, and sensitive information before publishing."
