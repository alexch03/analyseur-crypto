"""Le killswitch doit se déclencher sur les pertes réalisées, MÊME en mode paper.

Régression du bug observé : -598$ / 26 pertes consécutives sans déclenchement,
parce que les seuils n'étaient évalués que dans can_open() (court-circuité quand
mode == 'disabled')."""

from __future__ import annotations

import pytest

from app.execution import safety as safety_mod
from app.execution.safety import SafetyConfig, SafetyGuard


@pytest.fixture
def isolated_state(monkeypatch, tmp_path):
    """Isole le fichier d'état pour ne pas toucher au vrai data/safety_state.json."""
    monkeypatch.setattr(safety_mod, "STATE_FILE", tmp_path / "safety_state.json")


def test_killswitch_trips_on_consecutive_losses_in_paper_mode(isolated_state):
    guard = SafetyGuard(SafetyConfig(mode="disabled", max_consecutive_losses=5,
                                     max_daily_loss_usd=1e9))
    for _ in range(5):
        guard.record_close(pnl_usd=-1.0)
    assert guard.killswitch_tripped()
    assert "consecutive" in guard.state.killswitch_reason


def test_killswitch_trips_on_daily_loss_in_paper_mode(isolated_state):
    guard = SafetyGuard(SafetyConfig(mode="disabled", max_consecutive_losses=999,
                                     max_daily_loss_usd=200.0))
    guard.record_close(pnl_usd=-250.0)
    assert guard.killswitch_tripped()
    assert "daily loss" in guard.state.killswitch_reason


def test_win_resets_consecutive_losses(isolated_state):
    guard = SafetyGuard(SafetyConfig(mode="disabled", max_consecutive_losses=5,
                                     max_daily_loss_usd=1e9))
    for _ in range(4):
        guard.record_close(pnl_usd=-1.0)
    guard.record_close(pnl_usd=+2.0)         # gain -> reset compteur
    for _ in range(4):
        guard.record_close(pnl_usd=-1.0)
    assert not guard.killswitch_tripped()    # seulement 4 pertes consécutives après reset


def test_no_trip_below_thresholds(isolated_state):
    guard = SafetyGuard(SafetyConfig(mode="disabled", max_consecutive_losses=5,
                                     max_daily_loss_usd=200.0))
    for _ in range(3):
        guard.record_close(pnl_usd=-10.0)
    assert not guard.killswitch_tripped()
