# Methodology

The pipeline follows a *what-why-how* structure; section numbers refer to the manuscript.

## 1. Data and outcome (Section 3.2)
* 757 responses, 31 variables: `Gender, Job, Income, Area, Frequently` and 26 five-point Likert items
  (PU 4, PEU 4, FSC 3, SP 4, TP 3, IB 4, AUB 4). `Job` is constant and dropped -> **26 predictors**
  (4 demographics + 22 items). The AUB (actual use behaviour) items are **never** predictors.
* Target: `y = 1` if mean(AUB items) >= 4 (378 high / 379 other users). Code: `tg_stack.data.make_xy`.

## 2. Data-quality audit and leakage control (Section 3.3)
* Straight-liners (zero variance over all 26 Likert items: 158) and exact duplicates (129) are counted.
* Every distinct response vector gets a **group id** (628 groups). All outer splits and all inner tuning splits use
  `StratifiedGroupKFold`, so duplicated respondents never appear on both sides of a split.

## 3. Explanatory model (Section 3.4, stage `sem`)
Composite-based reliability (Cronbach's alpha, CR, AVE from one-factor loadings), HTMT discriminant validity, and OLS on
standardised composites with 5,000 bootstrap resamples:
`IB = b1 PU + b2 PEU + b3 FSC + b4 SP + b5 TP + e1`, `AUB = b6 IB + b7 FSC + b8 TP + e2`.

## 4. Theory-guided feature engineering (Section 3.5, `tg_stack.features.TheoryFeatures`)
26 raw predictors + 17 features = **43**:

| Group | Features | Theory |
|---|---|---|
| 6 composites | item means of PU, PEU, FSC, SP, TP, IB | measurement model |
| 6 dispersions | within-construct item SD | attitudinal ambivalence |
| 4 interactions | PU x PEU, IB x FSC, IB x TP, SP x TP | TAM; UTAUT; trust closes the intention-behaviour gap; S-O-R |
| 1 response style | SD over all 22 predictor items | down-weights straight-lined profiles |

The transformer is stateless and sits *inside* the sklearn pipeline, so it is recomputed within each fold.

## 5. TG-Stack (Section 3.6, `tg_stack.models.build_tg_stack`)
Level 0: logistic regression, RBF-SVM, Extra-Trees, CatBoost, LightGBM, MLP (fixed hyper-parameters, see Appendix
Table A1 of the paper and `models.base_learners`). Level 1: L2 logistic meta-learner on 5-fold out-of-fold
probabilities. Base learners are then refitted on the full training set.

## 6. Baselines and nested tuning (Section 3.7)
LR, NB, KNN, DT, SVM, MLP, RF, ET, AdaBoost, GBM, HistGB, XGBoost, LightGBM, CatBoost. Each is tuned **inside every
outer training fold** by randomised search (10 configurations, inner stratified group 3-fold, AUC criterion) over the
spaces in `models.baselines`.

## 7. Validation and statistics (Section 3.8)
* Outer loop: 5 x 5 repeated stratified group k-fold (seeds 0-4) -> 25 train/test splits per model.
* Metrics: accuracy, precision, recall, F1, ROC-AUC, MCC, Cohen's kappa, Brier score (threshold 0.5).
* Pairwise tests: Nadeau-Bengio corrected resampled t-test (`metrics.nb_ttest`).
* Global test: Friedman + Nemenyi critical difference (`metrics.friedman`).

## 8. Ablation, robustness, leakage (Section 3.9)
* 17 configurations (full + 16 variants): without theory features / interactions / dispersion, soft voting instead of the
  meta-learner, leave-one-learner-out (6), single base learner + features (6).
* R1: remove straight-liners and duplicates (n = 591). R2: outcome threshold 3.75. Compared with six baselines at default settings.
* Leakage: naive stratified k-fold vs group k-fold for TG-Stack, a default random forest, and a random-forest regression of continuous AUB.

## 9. Explainability (Section 3.10, stage `explain`)
Construct-level grouped permutation importance on held-out folds (all items of a construct permuted jointly, 10
repetitions), TreeSHAP on the CatBoost member, and the meta-learner weights - triangulated with the SEM paths.
