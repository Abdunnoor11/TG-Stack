import json

import pandas as pd
import pytest

from tg_stack.cli import build_parser, main
from tg_stack.evaluation import cv_eval
from tg_stack.models import baselines
from tg_stack.stages import ALL_STAGES, audit


def test_stage_registry_order():
    assert ALL_STAGES == ["audit", "sem", "main", "ablation", "robust", "explain", "leak", "figures"]


def test_parser_defaults_and_choices():
    a = build_parser().parse_args(["--data", "x.csv"])
    assert a.out == "results" and a.threads == 2 and not a.quick and not a.no_strict and a.stages is None
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--data", "x.csv", "--stages", "nonsense"])


def test_audit_stage_writes_json(ctx):
    audit.run(ctx)
    d = json.loads((ctx.out / "audit.json").read_text())
    assert d["audit"]["n"] == ctx.n and set(d["demographics"]) == {"Gender", "Income", "Area", "Frequently"}


def test_cli_end_to_end_audit_and_sem(csv_path, tmp_path, capsys):
    out = tmp_path / "res"
    rc = main(["--data", str(csv_path), "--out", str(out), "--stages", "audit", "--no-strict", "--quick"])
    assert rc == 0
    assert (out / "audit.json").exists()
    assert "stage: audit" in capsys.readouterr().out


def test_strict_mode_stops_on_fold_mismatch(csv_path, tmp_path):
    # synthetic data can never reproduce the paper's fold fingerprint
    with pytest.raises(SystemExit):
        main(["--data", str(csv_path), "--out", str(tmp_path / "r"), "--stages", "audit"])


@pytest.mark.slow
def test_cv_eval_runs_and_is_deterministic(ctx):
    bl = baselines(threads=1)
    models = {"LR": bl["LR"], "NB": bl["NB"]}
    a = cv_eval(models, ctx.X, ctx.y, ctx.groups, ctx.settings, ctx.out, tune=True, tag="t", save_oof=True)
    b = cv_eval(models, ctx.X, ctx.y, ctx.groups, ctx.settings, ctx.out, tune=True, tag="t2")
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 2 * ctx.settings.k * ctx.settings.repeats
    assert (ctx.out / "oof_t.npz").exists() and (ctx.out / "best_params_t.json").exists()
