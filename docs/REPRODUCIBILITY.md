# Reproducibility

## Environment used for the paper
Python 3.13.16 on a 2-vCPU Linux container (8 GB RAM), two threads per model, global seed 42.

| Package | Version | Package | Version |
|---|---|---|---|
| scikit-learn | **1.9.1** (critical) | numpy | 2.5.3 |
| xgboost | 3.4.1 | scipy | 1.18.1 |
| lightgbm | 4.7.0 | pandas | 3.0.5 |
| catboost | 1.2.10 | matplotlib | 3.11.2 |
| shap | 0.53.0 | | |

`pip install -r requirements-paper.txt` installs exactly this set.

## Seeds
Model seed 42; outer-fold seeds 0-4 (one per repetition); inner tuning-fold seeds 100-104; random-search sampler seed
42; bootstrap and permutation generators seeded with 42.

## Built-in guards
Before any modelling `tg-stack` checks

1. the installed versions of the four critical libraries (scikit-learn, xgboost, lightgbm, catboost) and reports the others;
2. the **MD5 fingerprint of the 25 outer test folds** (`1ae394ab002cf242dffaa3fc99bf2f46`), which verifies at once that
   the data file and the scikit-learn fold assignment are identical to the paper.

If either differs the run stops; `--no-strict` turns the errors into warnings (results then need not match the paper).

## Expected run time
About 2-2.5 hours for all stages on two CPU cores (`--quick` uses 1 x 5 folds for a smoke test).

## Verifying a reproduction
1. `TG_STACK_DATA=<csv> pytest -m paper` checks the audit counts (757 / 158 / 129 / 628 / 378 positives) and the fold fingerprint.
2. Compare `results/table_main.csv`, `table_ablation.csv`, `table_robust.csv`, ... with [`RESULTS.md`](RESULTS.md).

## Equivalence with the original script
`tg_stack` is a mechanical refactor of the single script `tg_stack_reproduce.py` used for the manuscript: models,
hyper-parameters, seeds, splits and statistics are unchanged. On a synthetic dataset with the real schema the original
script and this package produce identical output files (see `CHANGELOG.md`); only Figure 1's drawn numbers are now read
from the data instead of being hard-coded.

## Synthetic data
`data/synthetic_demo.csv` and `tg_stack.synthetic` are for tests/smoke runs only and are not real survey data.
