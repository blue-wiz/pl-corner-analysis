"""Generate every chart for the Premier League corner analysis.

    python -m src.main
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import analyse, ingest  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs"
INK, TEAL, SIGNAL, GREY = "#10202E", "#1E7A8C", "#E08A1E", "#9AA7B0"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("pl-corners")


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#C3CCD6")
    ax.grid(axis="x", color="#E9EDF1", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_attacking(att, path):
    top = att.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(top))
    ax.barh(y, top["goals"], color=TEAL, height=0.6)
    ax.set_yticks(y, top["team"])
    for i, (g, c) in enumerate(zip(top["goals"], top["corners"])):
        ax.text(g + 0.1, i, f"{g}  ({g/c:.1%} of {c})", va="center", fontsize=8, color=INK)
    ax.set_xlabel("Goals from corners")
    ax.set_title("Goals scored from corners, 2015/16", color=INK, fontsize=13, loc="left")
    _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_defending(dfn, path):
    top = dfn.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(top))
    ax.barh(y, top["goals_conceded"], color=SIGNAL, height=0.6)
    ax.set_yticks(y, top["team"])
    for i, (g, c) in enumerate(zip(top["goals_conceded"], top["corners_faced"])):
        ax.text(g + 0.1, i, f"{g}  ({g/c:.1%} of {c})", va="center", fontsize=8, color=INK)
    ax.set_xlabel("Goals conceded from corners")
    ax.set_title("Goals conceded from corners, 2015/16", color=INK, fontsize=13, loc="left")
    _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_first_contact(contact, path):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    order = contact.set_index("first_contact").reindex(["attacker", "defender", "none"]).reset_index()
    x = np.arange(len(order))
    ax.bar(x - 0.2, order["shot_rate"], width=0.38, color=TEAL, label="Shot rate")
    ax.bar(x + 0.2, order["goal_rate"], width=0.38, color=SIGNAL, label="Goal rate")
    ax.set_xticks(x, ["Attacker wins\nfirst contact", "Defender wins\nfirst contact", "No clean\ncontact"])
    for i, (s, g) in enumerate(zip(order["shot_rate"], order["goal_rate"])):
        ax.text(i - 0.2, s + 0.005, f"{s:.1%}", ha="center", fontsize=8, color=INK)
        ax.text(i + 0.2, g + 0.005, f"{g:.1%}", ha="center", fontsize=8, color=INK)
    ax.set_ylabel("Rate per corner")
    ax.set_title("First contact decides the corner", color=INK, fontsize=13, loc="left")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E9EDF1"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_why_scatter(corr, path):
    """The counterintuitive bit: team first-contact% barely predicts goal rate."""
    att = corr["attack_points"]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.scatter(att["first_contact_pct"], att["goal_rate"], s=54, color=TEAL, zorder=3)
    for team, row in att.iterrows():
        ax.annotate(team, (row["first_contact_pct"], row["goal_rate"]),
                    fontsize=7, color=INK, xytext=(4, 3), textcoords="offset points")
    # trend line
    x, y = att["first_contact_pct"], att["goal_rate"]
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 20)
    ax.plot(xs, a + b * xs, color=SIGNAL, linestyle="--", linewidth=1.4,
            label=f"r = {corr['attack_contact_vs_goal_rate_r']}")
    ax.set_xlabel("First-contact win rate")
    ax.set_ylabel("Goals per corner")
    ax.set_title("Winning first contact does NOT explain who scores", color=INK, fontsize=13, loc="left")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#E9EDF1"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_net(net, path):
    ordered = net.sort_values("net_corner_goals")
    fig, ax = plt.subplots(figsize=(9, 6.4))
    y = np.arange(len(ordered))
    colours = [TEAL if v >= 0 else SIGNAL for v in ordered["net_corner_goals"]]
    ax.barh(y, ordered["net_corner_goals"], color=colours, height=0.62)
    ax.set_yticks(y, ordered["team"])
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_xlabel("Net goals from corners (scored − conceded)")
    ax.set_title("Who wins the corner battle over a season", color=INK, fontsize=13, loc="left")
    _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    log.info("Loading corners (cached after first run)...")
    corners = ingest.add_zones(ingest.fetch_season_corners())
    log.info("%s corners across the season", f"{len(corners):,}")

    att = analyse.attacking_summary(corners)
    dfn = analyse.defending_summary(corners)
    contact = analyse.contact_breakdown(corners)
    zones = analyse.zone_breakdown(corners)
    delivery = analyse.delivery_breakdown(corners)
    corr = analyse.contact_vs_outcome_correlation(att, dfn)
    net = analyse.net_set_piece_table(att, dfn)

    plot_attacking(att, OUT / "attacking.png")
    plot_defending(dfn, OUT / "defending.png")
    plot_first_contact(contact, OUT / "first_contact.png")
    plot_why_scatter(corr, OUT / "why_scatter.png")
    plot_net(net, OUT / "net_corner_goals.png")

    for name, df in [("attacking", att), ("defending", dfn), ("zones", zones),
                     ("delivery", delivery), ("net", net)]:
        df.to_csv(OUT / f"{name}.csv", index=False)

    print(f"\nTop scorer from corners : {att.iloc[0]['team']} ({att.iloc[0]['goals']} goals)")
    print(f"Most conceded           : {dfn.iloc[0]['team']} ({dfn.iloc[0]['goals_conceded']})")
    print(f"Attacker first contact  : {contact.iloc[0]['goal_rate']:.1%} goal rate "
          f"vs {contact[contact.first_contact=='defender'].iloc[0]['goal_rate']:.1%} for defenders")
    print(f"Why r (attack)          : {corr['attack_contact_vs_goal_rate_r']}")
    print(f"Charts + tables written to {OUT}")


if __name__ == "__main__":
    main()
