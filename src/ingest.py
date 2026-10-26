"""Fetch every corner kick from a full StatsBomb-covered season.

Uses StatsBomb's open data (free, public, released under their user agreement).
The default competition is the 2015/16 Premier League — the one complete
English top-flight season StatsBomb make freely available with full event
detail. That is what makes team-by-team first-contact, zone and delivery
analysis possible: those fields do not exist in any free source for the current
season, which is licensed data.

Match event feeds are cached to data/ after the first download so re-runs are
fast and work offline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

OPEN_DATA_ROOT = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
REQUEST_TIMEOUT = 60

# 2015/16 Premier League.
DEFAULT_COMPETITION = 2
DEFAULT_SEASON = 27

GOAL_CENTRE = (120.0, 40.0)
CONTACT_EVENTS = {"Ball Receipt*", "Clearance", "Shot", "Duel", "Miscontrol", "Interception", "Block"}


@dataclass(frozen=True)
class Corner:
    """One corner kick, the team taking it, and what came of it."""

    match_id: int
    attacking_team: str
    defending_team: str
    minute: int
    end_x: float
    end_y: float
    delivery: str
    first_contact: str  # "attacker", "defender" or "none"
    shot_taken: bool
    goal: bool


def fetch_season_corners(
    competition_id: int = DEFAULT_COMPETITION,
    season_id: int = DEFAULT_SEASON,
    max_matches: int | None = None,
) -> pd.DataFrame:
    """Return one row per corner for the whole season.

    Downloads match by match, caching each event feed. `max_matches` caps the
    number of matches (useful for a quick smoke test); leave it None for the
    full season.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    matches = _load_json(f"{OPEN_DATA_ROOT}/matches/{competition_id}/{season_id}.json",
                         CACHE_DIR / f"matches_{competition_id}_{season_id}.json")

    match_index = {m["match_id"]: m for m in matches}
    match_ids = list(match_index)
    if max_matches is not None:
        match_ids = match_ids[:max_matches]

    rows: list[dict] = []
    for i, match_id in enumerate(match_ids, 1):
        events = _load_json(
            f"{OPEN_DATA_ROOT}/events/{match_id}.json",
            CACHE_DIR / f"events_{match_id}.json",
        )
        rows.extend(_extract_corners(events, match_id))
        if i % 50 == 0:
            log.info("processed %s / %s matches", i, len(match_ids))

    if not rows:
        raise ValueError("no corners found")

    return pd.DataFrame(rows)


def _load_json(url: str, cache_path: Path) -> list | dict:
    """Fetch JSON, using a local cache when present."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    response = requests.get(url, headers={"Accept": "application/json"}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    cache_path.write_text(json.dumps(payload))
    return payload


def _extract_corners(events: list[dict], match_id: int) -> list[dict]:
    """Pull corner passes out of one match's event feed.

    Outcomes are linked by StatsBomb's `possession` counter, not by event
    adjacency: the event straight after a corner is nearly always a ball
    receipt or clearance, so adjacency finds no shots. The defending team is
    whichever team is not taking the corner in that possession.
    """
    teams = list({e.get("team", {}).get("name") for e in events if e.get("team")})
    corners = []

    for index, event in enumerate(events):
        pass_data = event.get("pass") or {}
        if pass_data.get("type", {}).get("name") != "Corner":
            continue

        attacking = event.get("team", {}).get("name")
        defending = next((t for t in teams if t and t != attacking), None)
        possession = event.get("possession")
        end_location = pass_data.get("end_location") or [None, None]

        followers = [e for e in events[index + 1 :] if e.get("possession") == possession]
        shots = [
            e for e in followers
            if e.get("type", {}).get("name") == "Shot"
            and e.get("team", {}).get("name") == attacking
        ]

        corners.append(
            {
                "match_id": match_id,
                "attacking_team": attacking,
                "defending_team": defending,
                "minute": event.get("minute"),
                "end_x": end_location[0],
                "end_y": end_location[1],
                "delivery": _classify_delivery(pass_data),
                "first_contact": _classify_contact(attacking, followers),
                "shot_taken": bool(shots),
                "goal": any(
                    (s.get("shot") or {}).get("outcome", {}).get("name") == "Goal" for s in shots
                ),
            }
        )

    return corners


def _classify_delivery(pass_data: dict) -> str:
    """Inswinger / outswinger where StatsBomb records it, else flight height."""
    technique = (pass_data.get("technique") or {}).get("name")
    if technique == "Inswinging":
        return "inswinger"
    if technique == "Outswinging":
        return "outswinger"
    if technique == "Straight":
        return "straight"

    height = (pass_data.get("height") or {}).get("name")
    if height == "High Pass":
        return "high (swing unknown)"
    if height in {"Low Pass", "Ground Pass"}:
        return "low/short"
    return "unclassified"


def _classify_contact(attacking: str, followers: list[dict]) -> str:
    """Who touched the ball first after the delivery."""
    for event in followers:
        if event.get("type", {}).get("name") in CONTACT_EVENTS:
            return "attacker" if event.get("team", {}).get("name") == attacking else "defender"
    return "none"


def add_zones(frame: pd.DataFrame) -> pd.DataFrame:
    """Label where each delivery landed, in zones coaches use."""
    out = frame.dropna(subset=["end_x", "end_y"]).copy()
    out["distance_to_goal"] = np.hypot(GOAL_CENTRE[0] - out["end_x"], GOAL_CENTRE[1] - out["end_y"])

    conditions = [
        (out["end_x"] >= 114) & (out["end_y"] < 36),
        (out["end_x"] >= 114) & (out["end_y"].between(36, 44)),
        (out["end_x"] >= 114) & (out["end_y"] > 44),
    ]
    out["zone"] = np.select(conditions, ["near post", "central six-yard", "far post"],
                            default="deeper / edge of area")
    return out
