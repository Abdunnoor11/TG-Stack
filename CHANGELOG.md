# Changelog

## 1.0.0 - 2026-10-16
- First public release: modular package (`tg_stack`) and CLI (`tg-stack`) refactored from the single
  reproduction script `tg_stack_reproduce.py` used for the manuscript. No change to the models, protocol or
  random seeds.
- Added test-suite, CI, synthetic demo data, documentation.
- Figure 1 now takes the numbers it draws (sample size, straight-liners, duplicates, groups, class counts,
  feature counts, SEM R^2) from the data/results instead of hard-coding them; figures whose inputs are missing
  are skipped with a warning instead of aborting the run.
