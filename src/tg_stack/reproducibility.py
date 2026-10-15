"""Guards that make sure a run reproduces the paper's numbers."""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import logging

from .evaluation import outer_folds

logger = logging.getLogger(__name__)

#: Library versions used for the paper (Python 3.13.16, 2-vCPU Linux container).
PAPER_VERSIONS = {
    "scikit-learn": "1.9.1",
    "xgboost": "3.4.1",
    "lightgbm": "4.7.0",
    "catboost": "1.2.10",
    "shap": "0.53.0",
    "numpy": "2.5.3",
    "scipy": "1.18.1",
    "pandas": "3.0.5",
    "matplotlib": "3.11.2",
}
#: Versions that change the cross-validated numbers; the others only cause a notice.
CRITICAL = {"scikit-learn", "xgboost", "lightgbm", "catboost"}
#: MD5 fingerprint of the 25 outer test folds (5 repeats x 5 folds, seeds 0-4) on the real data.
PAPER_FOLD_MD5 = "1ae394ab002cf242dffaa3fc99bf2f46"


class ReproducibilityError(SystemExit):
    """Raised (as a clean exit) when the environment or the folds differ from the paper."""


def check_versions(strict: bool = True) -> list:
    """Print installed vs. required versions; abort on critical mismatches if ``strict``."""
    bad = []
    for pkg, want in PAPER_VERSIONS.items():
        try:
            have = md.version(pkg)
        except md.PackageNotFoundError:
            have = "not installed"
        flag = "" if have == want else ("   <-- MISMATCH" if pkg in CRITICAL else "   (minor difference)")
        print(f"  {pkg:13s} required {want:8s} installed {have}{flag}")
        if have != want and pkg in CRITICAL:
            bad.append(pkg)
    if bad:
        msg = (
            "Version mismatch for " + ", ".join(bad) + ". Cross-validated results will NOT match the paper. "
            "Install the pinned versions (see requirements-paper.txt)."
        )
        if strict:
            raise ReproducibilityError("ERROR: " + msg + " Use --no-strict to run anyway.")
        logger.warning("WARNING: %s", msg)
    return bad


def fold_fingerprint(X, y, groups) -> str:
    """MD5 over the sorted test indices of all 5 x 5 outer folds."""
    folds = []
    for r in range(5):
        for _tr, te in outer_folds(X, y, groups, r, 5):
            folds.append(sorted(te.tolist()))
    return hashlib.md5(json.dumps(folds).encode()).hexdigest()


def check_folds(X, y, groups, strict: bool = True) -> str:
    """Compare the outer-fold fingerprint with the paper's."""
    md5 = fold_fingerprint(X, y, groups)
    same = md5 == PAPER_FOLD_MD5
    print(f"Outer-fold fingerprint {md5} ({'identical to the paper' if same else 'DIFFERENT from the paper'})")
    if not same and strict:
        raise ReproducibilityError(
            "ERROR: outer folds differ from the manuscript (check the data file and scikit-learn 1.9.1)."
            " Use --no-strict to run anyway."
        )
    return md5
