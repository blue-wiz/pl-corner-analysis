"""Tests for the PL corner analysis. Uses cached season if present, else synthetic."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src import analyse, ingest


@pytest.fixture(scope="module")
def corners():
    cache = Path(__file__).resolve().parent.parent / "data" / "corners.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rng = np.random.default_rng(0)
    n = 2000
    teams = [f"Team {i}" for i in range(20)]
    fc = rng.choice(["attacker", "defender", "none"], n, p=[0.63, 0.28, 0.09])
    shot = ((fc == "attacker") & (rng.random(n) < 0.47)) | ((fc == "defender") & (rng.random(n) < 0.20))
    goal = shot & (rng.random(n) < 0.10)
    return pd.DataFrame({
        "attacking_team": rng.choice(teams, n), "defending_team": rng.choice(teams, n),
        "first_contact": fc, "shot_taken": shot, "goal": goal,
        "delivery": rng.choice(["inswinger", "outswinger"], n),
        "zone": rng.choice(["near post", "central six-yard", "far post", "deeper / edge of area"], n),
        "end_x": rng.uniform(100, 120, n), "end_y": rng.uniform(30, 50, n),
    })


def test_wilson_brackets_estimate():
    low, high = analyse.wilson_interval([5, 50], [10, 100])
    assert low[0] <= 0.5 <= high[0]


def test_attacking_counts_reconcile(corners):
    att = analyse.attacking_summary(corners)
    assert att["corners"].sum() == len(corners)
    assert att["goals"].sum() == int(corners["goal"].sum())


def test_defending_counts_reconcile(corners):
    dfn = analyse.defending_summary(corners)
    assert dfn["corners_faced"].sum() == len(corners)


def test_first_contact_ordering(corners):
    c = analyse.contact_breakdown(corners).set_index("first_contact")
    assert c.loc["attacker", "goal_rate"] > c.loc["defender", "goal_rate"]
    if "none" in c.index:
        assert c.loc["none", "goal_rate"] == 0.0


def test_zone_labels_valid(corners):
    z = analyse.zone_breakdown(corners)
    assert set(z["zone"]) <= {"near post", "central six-yard", "far post", "deeper / edge of area"}


def test_delivery_classifier_swing():
    assert ingest._classify_delivery({"technique": {"name": "Inswinging"}}) == "inswinger"
    assert ingest._classify_delivery({"technique": {"name": "Outswinging"}}) == "outswinger"
    assert ingest._classify_delivery({"height": {"name": "High Pass"}}) == "high (swing unknown)"


def test_first_contact_classifier():
    assert ingest._classify_contact("A", [{"type": {"name": "Clearance"}, "team": {"name": "B"}}]) == "defender"
    assert ingest._classify_contact("A", [{"type": {"name": "Shot"}, "team": {"name": "A"}}]) == "attacker"
    assert ingest._classify_contact("A", [{"type": {"name": "Pressure"}}]) == "none"
