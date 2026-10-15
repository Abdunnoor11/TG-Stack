# Known limitations and caveats

1. **Stacking's internal split is not group-aware.** `StackingClassifier(cv=5)` creates its out-of-fold probabilities
   with a stratified (not group) 5-fold split. This reproduces the paper's results, but duplicated respondents can fall
   on both sides of that internal split, so the meta-learner's weights (e.g. the large Extra-Trees weight) may be
   optimistic for learners that memorise. Outer-fold test estimates are not affected. *Possible fix:* pass a
   group-aware iterable of splits to `StackingClassifier(cv=...)` from a wrapper that knows the training groups
   (changes the published numbers; evaluate as a new experiment).
2. **TG-Stack's hyper-parameters are fixed by hand**, not selected inside the nested loop. Baselines in the robustness
   scenarios R1/R2 are untuned, so margins there are not comparable with the nested-tuned benchmark.
3. **Friedman-Nemenyi** treats the 25 folds of a repeated CV as independent blocks; read the ranking descriptively. The
   Nadeau-Bengio tests are the primary inference.
4. **scikit-learn deprecation:** `SVC(probability=True)` emits a `FutureWarning` in 1.9 and the parameter is scheduled for
   removal in 1.11. The pinned environment is unaffected; with newer scikit-learn replace it by
   `CalibratedClassifierCV(SVC(), ensemble=False)` (this changes the SVM probabilities).
5. **Data:** cross-sectional, self-reported, one country, strongly female (77.7%) and low-income (88.8%) sample;
   high HTMT values and 21% straight-lining indicate common-method variance. Binarising AUB discards information.
6. The SEM uses equally weighted composites with bootstrapping, not the iterative PLS algorithm, so coefficients may
   differ slightly from SmartPLS.
