"""Team-level corner analysis for a full season.

Answers the questions a set-piece coach actually asks:
  - Which teams score most from corners, and which concede most?
  - When a corner comes in, who wins the first contact?
  - Does winning first contact explain scoring and conceding? (the "why")
  - Which deliveries and which landing zones convert?

Everything is reported with Wilson confidence intervals where it's a rate,
because a team takes a few hundred corners in a season and the differences
between mid-table sides are often inside the noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wilson_interval(successes, trials, z: float = 1.96):
    """95% Wilson score interval for a proportion. Vectorised."""
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(trials > 0, successes / trials, 0.0)
        denom = 1 + z**2 / trials
        centre = (p + z**2 / (2 * trials)) / denom
        margin = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return np.nan_to_num(centre - margin), np.nan_to_num(centre + margin)


def attacking_summary(corners: pd.DataFrame) -> pd.DataFrame:
    """Per team: corners taken, first-contact win rate, shots, goals."""
    grouped = corners.groupby("attacking_team").agg(
        corners=("goal", "size"),
        first_contact_won=("first_contact", lambda s: (s == "attacker").sum()),
        shots=("shot_taken", "sum"),
        goals=("goal", "sum"),
    )
    grouped["first_contact_pct"] = (grouped["first_contact_won"] / grouped["corners"]).round(3)
    grouped["shot_rate"] = (grouped["shots"] / grouped["corners"]).round(3)
    grouped["goal_rate"] = (grouped["goals"] / grouped["corners"]).round(4)

    low, high = wilson_interval(grouped["goals"], grouped["corners"])
    grouped["goal_rate_low"] = low.round(4)
    grouped["goal_rate_high"] = high.round(4)

    return grouped.sort_values("goals", ascending=False).reset_index().rename(
        columns={"attacking_team": "team"}
    )


def defending_summary(corners: pd.DataFrame) -> pd.DataFrame:
    """Per team: corners faced, how often they win first contact, goals conceded."""
    grouped = corners.groupby("defending_team").agg(
        corners_faced=("goal", "size"),
        defensive_contact_won=("first_contact", lambda s: (s == "defender").sum()),
        shots_conceded=("shot_taken", "sum"),
        goals_conceded=("goal", "sum"),
    )
    grouped["defensive_contact_pct"] = (
        grouped["defensive_contact_won"] / grouped["corners_faced"]
    ).round(3)
    grouped["concede_rate"] = (grouped["goals_conceded"] / grouped["corners_faced"]).round(4)

    low, high = wilson_interval(grouped["goals_conceded"], grouped["corners_faced"])
    grouped["concede_rate_low"] = low.round(4)
    grouped["concede_rate_high"] = high.round(4)

    return grouped.sort_values("goals_conceded", ascending=False).reset_index().rename(
        columns={"defending_team": "team"}
    )


def contact_breakdown(corners: pd.DataFrame) -> pd.DataFrame:
    """League-wide: what happens after each first-contact outcome."""
    return (
        corners.groupby("first_contact")
        .agg(
            corners=("goal", "size"),
            shot_rate=("shot_taken", "mean"),
            goal_rate=("goal", "mean"),
        )
        .round(4)
        .sort_values("goal_rate", ascending=False)
        .reset_index()
    )


def zone_breakdown(corners: pd.DataFrame) -> pd.DataFrame:
    """League-wide conversion by landing zone, with Wilson intervals."""
    grouped = corners.groupby("zone").agg(
        corners=("goal", "size"),
        shots=("shot_taken", "sum"),
        goals=("goal", "sum"),
    )
    grouped["shot_rate"] = (grouped["shots"] / grouped["corners"]).round(3)
    grouped["goal_rate"] = (grouped["goals"] / grouped["corners"]).round(4)
    low, high = wilson_interval(grouped["shots"], grouped["corners"])
    grouped["shot_rate_low"] = low.round(3)
    grouped["shot_rate_high"] = high.round(3)
    return grouped.sort_values("shot_rate", ascending=False).reset_index()


def delivery_breakdown(corners: pd.DataFrame) -> pd.DataFrame:
    """League-wide conversion by delivery type."""
    return (
        corners.groupby("delivery")
        .agg(
            corners=("goal", "size"),
            first_contact_pct=("first_contact", lambda s: (s == "attacker").mean()),
            shot_rate=("shot_taken", "mean"),
            goal_rate=("goal", "mean"),
        )
        .round(4)
        .sort_values("goal_rate", ascending=False)
        .reset_index()
    )


def contact_vs_outcome_correlation(attack: pd.DataFrame, defend: pd.DataFrame) -> dict:
    """Does winning first contact explain scoring and conceding?

    This is the 'why' the question asks for. Returns Pearson correlations
    across the 20 teams, plus the underlying series for plotting.
    """
    att = attack.set_index("team")
    dfn = defend.set_index("team")

    attack_r = np.corrcoef(att["first_contact_pct"], att["goal_rate"])[0, 1]
    defend_r = np.corrcoef(dfn["defensive_contact_pct"], dfn["concede_rate"])[0, 1]

    return {
        "attack_contact_vs_goal_rate_r": round(float(attack_r), 3),
        "defend_contact_vs_concede_rate_r": round(float(defend_r), 3),
        "attack_points": att[["first_contact_pct", "goal_rate"]],
        "defend_points": dfn[["defensive_contact_pct", "concede_rate"]],
    }


def net_set_piece_table(attack: pd.DataFrame, defend: pd.DataFrame) -> pd.DataFrame:
    """Goals from corners minus goals conceded from corners, per team."""
    merged = attack[["team", "goals", "corners"]].merge(
        defend[["team", "goals_conceded", "corners_faced"]], on="team"
    )
    merged["net_corner_goals"] = merged["goals"] - merged["goals_conceded"]
    return merged.sort_values("net_corner_goals", ascending=False).reset_index(drop=True)
