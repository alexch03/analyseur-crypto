from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.services import optimization_batch as ob


def test_slug_for_autoopt_filename_fallback():
    s = ob.slug_for_autoopt_filename(None, "net_pnl_quote", "exhaustive", 3)
    assert "net_pnl_quote" in s
    assert "exhaustive" in s
    assert "j3" in s


def test_slug_for_autoopt_label_sanitized():
    s = ob.slug_for_autoopt_filename("  Ma Run #1  ", "composite", "random", 1)
    assert " " not in s
    assert s.startswith("ma_run")


def test_allocate_autoopt_output_filename_increments(tmp_path: Path) -> None:
    with patch.object(ob, "workspace_methods_dir", return_value=tmp_path):
        n1 = ob.allocate_autoopt_output_filename("myMethod.json", "obj_a")
        assert n1 == "myMethod__autoopt__obj_a.json"
        (tmp_path / n1).write_text("{}")
        n2 = ob.allocate_autoopt_output_filename("myMethod.json", "obj_a")
        assert n2 == "myMethod__autoopt__obj_a__002.json"


def test_build_effective_state_for_batch_job_overrides_objective(tmp_path: Path, monkeypatch) -> None:
    from app.services import dashboard_workspace as dw

    name = "tmp_batch_method.json"
    (tmp_path / name).write_text(
        '{"optimization_objective": "net_r", "optimization_strategy": "exhaustive"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(dw, "workspace_methods_dir", lambda: tmp_path)

    st = {
        "symbols": ["BTC/USDT"],
        "optimization_objective": "net_pnl_quote",
        "optimization_strategy": "exhaustive",
        "optimization_max_trials": 200,
        "optimization_grid": {},
        "best_engine_params": {},
        "dashboard_method_file": None,
    }
    eff = ob.build_effective_state_for_batch_job(
        st,
        source_method=name,
        optimization_objective="penalized_pnl_quote",
        optimization_strategy="random",
        optimization_max_trials=99,
        optimization_grid=None,
    )
    assert eff["optimization_objective"] == "penalized_pnl_quote"
    assert eff["optimization_strategy"] == "random"
    assert eff["optimization_max_trials"] == 99
