"""Figures built from stage outputs. Every plotted number is read from outputs/, never typed in."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import OUTPUTS  # noqa: E402
from .io import read_json  # noqa: E402


def apply_style(style: dict[str, Any]) -> None:
    plt.rcParams.update(
        {
            "font.family": style["font_family"],
            "font.size": style["font_size"],
            "axes.titlesize": style["title_size"],
            "axes.labelsize": style["label_size"],
            "xtick.labelsize": style["tick_size"],
            "ytick.labelsize": style["tick_size"],
            "axes.edgecolor": style["axis"],
            "axes.linewidth": 0.8,
            "xtick.color": style["text_secondary"],
            "ytick.color": style["text_secondary"],
            "axes.labelcolor": style["text_primary"],
            "text.color": style["text_primary"],
            "figure.facecolor": style["surface"],
            "axes.facecolor": style["surface"],
            "savefig.facecolor": style["surface"],
            "axes.grid": False,
            "legend.frameon": False,
            "legend.fontsize": style["font_size"],
            "svg.fonttype": "none",
        }
    )


def _tidy(ax: plt.Axes, style: dict[str, Any]) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=0.6, color=style["axis"])


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width))


def _save(fig: plt.Figure, out_dir: Path, name: str, cfg: dict[str, Any]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in cfg["output"]["formats"]:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=cfg["output"]["dpi"], format=fmt, bbox_inches="tight", pad_inches=0.08)
        paths.append(path)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------- figure 1


def figure1_data() -> dict[str, Any]:
    frame = pd.read_csv(OUTPUTS / "estimands" / "estimands.csv")
    repro = read_json(OUTPUTS / "reproduce" / "summary.json")
    conf = repro["runs"]["confirmatory"]
    groups = ("genpop", "unknown_ai", "famous_ai", "famous_nonai")
    n_rows = {run: int(sum(repro["runs"][run]["counts"][g]["valid_rows"] for g in groups)) for run in ("discovery", "confirmatory")}
    return {
        "estimands": frame.to_dict(orient="records"),
        "response_level_sd_pp": conf["response_level_sd_pp"],
        "interaction_in_response_sd": conf["interaction_in_response_sd"],
        "reps": repro["reps"],
        "n_valid_rows_primary_groups": n_rows,
        "n_identities_per_group": 70,
    }


def figure1(cfg: dict[str, Any], out_dir: Path) -> tuple[list[Path], dict[str, Any], str]:
    style, pal, f1 = cfg["style"], cfg["palette"], cfg["figure1"]
    data = figure1_data()
    frame = pd.DataFrame(data["estimands"])
    order = f1["estimand_order"]
    fig, ax = plt.subplots(figsize=tuple(f1["size_in"]))
    y_base = np.arange(len(order))[::-1]
    handles = {}
    markers = {"discovery": "s", "confirmatory": "o"}
    for k, run in enumerate(("discovery", "confirmatory")):
        color = pal[run]
        offset = f1["run_offset"] * (0.5 - k)
        for i, name in enumerate(order):
            row = frame[(frame.run == run) & (frame.estimand == name)].iloc[0]
            y = y_base[i] + offset
            ax.errorbar(
                row.point_pp,
                y,
                xerr=[[row.point_pp - row.ci95_lower_pp], [row.ci95_upper_pp - row.point_pp]],
                fmt="none",
                ecolor=color,
                elinewidth=style["line_width"],
                capsize=style["cap_size"],
                capthick=style["line_width"] * 0.7,
                zorder=2,
            )
            (h,) = ax.plot(row.point_pp, y, marker=markers[run], ms=style["marker_size"], color=color, mec=style["surface"], mew=style["surface_ring"], ls="none", zorder=3)
            handles[run] = h
    ax.axvline(0, color=style["zero_line"], lw=0.8, zorder=1)
    ax.set_yticks(y_base)
    ax.set_yticklabels([f1["estimand_labels"][n] for n in order], color=style["text_primary"])
    ax.set_xlabel(f1["x_label"])
    ax.grid(axis="x", color=style["grid"], lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.legend(
        [handles["discovery"], handles["confirmatory"]],
        [f1["run_labels"]["discovery"], f1["run_labels"]["confirmatory"]],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    ax.set_title(_wrap(f1["title"], 88), loc="left", pad=10)
    _tidy(ax, style)
    inter = frame[(frame.run == "confirmatory") & (frame.estimand == "interaction")].iloc[0]
    caption = f1["caption_template"].format(
        reps=data["reps"],
        interaction=inter.point_pp,
        sd_units=abs(data["interaction_in_response_sd"]),
        sd=data["response_level_sd_pp"],
        n_rows_disc=data["n_valid_rows_primary_groups"]["discovery"],
        n_rows_conf=data["n_valid_rows_primary_groups"]["confirmatory"],
    )
    fig.text(-0.3, -0.2, _wrap(caption, 125), ha="left", va="top", fontsize=style["annotation_size"], color=style["text_primary"], transform=ax.transAxes)
    paths = _save(fig, out_dir, f1["file"], cfg)
    return paths, data, caption


# ---------------------------------------------------------------- figure 2


def figure2_data() -> dict[str, Any]:
    table = pd.read_csv(OUTPUTS / "identities" / "identities.csv")
    table = table[table.profile == "constructed"].copy()
    summary = read_json(OUTPUTS / "identities" / "summary.json")
    merged_path = OUTPUTS / "roles" / "merged_coding.csv"
    roles = None
    if merged_path.exists():
        merged = pd.read_csv(merged_path, keep_default_na=False)
        if (merged["final_category"].astype(str).str.strip() == "").any():
            roles = None
        else:
            roles = dict(zip(merged.persona_key, merged.final_category))
    records = table[["persona_key", "name", "discovery_twin_adjusted_pp", "confirmatory_twin_adjusted_pp"]].to_dict(orient="records")
    for r in records:
        r["role"] = roles.get(r["persona_key"]) if roles else None
    return {
        "identities": records,
        "spearman": summary["spearman_discovery_vs_confirmatory_twin_adjusted"],
        "role_coding_available": roles is not None,
    }


def _role_centroids(frame: pd.DataFrame, reps: int, seed: int) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out = {}
    for role, part in frame.groupby("role"):
        x = part.discovery_twin_adjusted_pp.to_numpy()
        y = part.confirmatory_twin_adjusted_pp.to_numpy()
        n = len(part)
        draws = np.array([[x[idx].mean(), y[idx].mean()] for idx in (rng.integers(0, n, size=n) for _ in range(reps))])
        out[str(role)] = {
            "n": int(n),
            "x": float(x.mean()),
            "y": float(y.mean()),
            "x_ci95": [float(v) for v in np.percentile(draws[:, 0], [2.5, 97.5])],
            "y_ci95": [float(v) for v in np.percentile(draws[:, 1], [2.5, 97.5])],
        }
    return out


def _label_points(ax: plt.Axes, labelled: pd.DataFrame, all_points: pd.DataFrame, style: dict[str, Any], leader_color: str) -> None:
    """Identity labels repelled from every point and from each other, with thin connectors."""
    from adjustText import adjust_text

    x_col, y_col = "discovery_twin_adjusted_pp", "confirmatory_twin_adjusted_pp"
    texts = [
        ax.text(row[x_col], row[y_col], row["name"], fontsize=style["annotation_size"] - 0.5, color=style["text_primary"], zorder=6)
        for _, row in labelled.iterrows()
    ]
    adjust_text(
        texts,
        x=all_points[x_col].to_numpy(),
        y=all_points[y_col].to_numpy(),
        ax=ax,
        expand=(1.6, 1.9),
        force_text=(0.7, 1.0),
        force_points=(0.9, 1.1),
        force_static=(0.9, 1.1),
        only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
        arrowprops={"arrowstyle": "-", "color": leader_color, "lw": 0.8, "shrinkA": 0, "shrinkB": 3},
        min_arrow_len=4,
    )


def figure2(cfg: dict[str, Any], out_dir: Path) -> tuple[list[Path], dict[str, Any], str]:
    style, pal, f2 = cfg["style"], cfg["palette"], cfg["figure2"]
    data = figure2_data()
    frame = pd.DataFrame(data["identities"])
    fig, ax = plt.subplots(figsize=tuple(f2["size_in"]))
    coded = data["role_coding_available"]
    if coded:
        colors = frame.role.map(pal["roles"]).fillna(pal["roles"]["other"])
    else:
        colors = pal["uniform"]
    ax.axhline(0, color=style["zero_line"], lw=0.7, zorder=1)
    ax.axvline(0, color=style["zero_line"], lw=0.7, zorder=1)
    lim = float(np.nanmax(np.abs(frame[["discovery_twin_adjusted_pp", "confirmatory_twin_adjusted_pp"]].to_numpy()))) * 1.15
    if f2.get("identity_line"):
        ax.plot([-lim, lim], [-lim, lim], color=style["axis"], lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.text(lim * 0.97, lim * 0.97, "y = x", ha="right", va="bottom", fontsize=style["annotation_size"] - 1, color=style["text_secondary"], rotation=45, rotation_mode="anchor")
    ax.scatter(frame.discovery_twin_adjusted_pp, frame.confirmatory_twin_adjusted_pp, s=f2["point_size"], c=colors, alpha=float(f2.get("point_alpha", 1.0)), edgecolors=style["surface"], linewidths=0.8, zorder=3)
    centroids = {}
    if coded:
        centroids = _role_centroids(frame, int(f2["role_bootstrap_reps"]), int(f2["role_bootstrap_seed"]))
        for role, c in centroids.items():
            color = pal["roles"].get(role, pal["roles"]["other"])
            ax.plot(c["x_ci95"], [c["y"], c["y"]], color=color, lw=1.6, zorder=4)
            ax.plot([c["x"], c["x"]], c["y_ci95"], color=color, lw=1.6, zorder=4)
            ax.scatter([c["x"]], [c["y"]], s=f2["centroid_size"], c=color, marker="D", edgecolors=style["surface"], linewidths=1.2, zorder=5, label=f"{role} (n={c['n']})")
        ax.legend(loc="lower right", title=None)
    data["role_centroids"] = centroids
    frame["extreme"] = np.hypot(frame.discovery_twin_adjusted_pp, frame.confirmatory_twin_adjusted_pp)
    labelled = frame.nlargest(int(f2["n_labels"]), "extreme")
    data["labelled_identities"] = labelled.persona_key.tolist()
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    _label_points(ax, labelled, frame, style, f2.get("leader_color", style["axis"]))
    rho = data["spearman"]
    ax.text(0.02, 0.97, f2["rho_template"].format(rho=rho["rho"], lo=rho["ci95"][0], hi=rho["ci95"][1]) + f"\n(n = {rho['n_identities']} identities)", transform=ax.transAxes, ha="left", va="top", fontsize=style["annotation_size"], color=style["text_primary"], bbox={"boxstyle": "round,pad=0.3", "fc": style["surface"], "ec": style["grid"]})
    ax.set_aspect("equal")
    ax.set_xlabel(f2["x_label"])
    ax.set_ylabel(f2["y_label"])
    title = f2["title_template"].format(rho=rho["rho"], lo=rho["ci95"][0], hi=rho["ci95"][1])
    ax.set_title(_wrap(title, 66) + "\n" + f2["subtitle"], loc="left", pad=10)
    ax.grid(color=style["grid"], lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    _tidy(ax, style)
    caption = f2["coded_caption"] if coded else f2["pending_caption"]
    fig.text(0.0, -0.13, _wrap(caption, 95), ha="left", va="top", fontsize=style["annotation_size"], color=style["text_secondary"], transform=ax.transAxes)
    paths = _save(fig, out_dir, f2["file"], cfg)
    return paths, data, caption


# ---------------------------------------------------------------- figure 3


def figure3_data() -> dict[str, Any]:
    d = read_json(OUTPUTS / "decompose" / "summary.json")
    return {
        "yes_rate": {"point": d["yes_rate_interaction"]["interaction"], "ci95": d["yes_rate_interaction"]["ci95"]},
        "folded": {"point": d["folded_confidence_interaction"]["interaction"], "ci95": d["folded_confidence_interaction"]["ci95"]},
        "standardized": {"point": d["choice_standardized_confidence_interaction"]["interaction_pp"], "ci95": d["choice_standardized_confidence_interaction"]["ci95_pp"]},
        "matched": {"point": d["same_choice_matched"]["interaction_pp"], "ci95": d["same_choice_matched"]["ci95_pp"]},
        "reps": d["reps"],
    }


def figure3(cfg: dict[str, Any], out_dir: Path) -> tuple[list[Path], dict[str, Any], str]:
    style, pal, f3 = cfg["style"], cfg["palette"], cfg["figure3"]
    data = figure3_data()
    panels = f3["panels"]
    fig, axes = plt.subplots(1, len(panels), figsize=tuple(f3["size_in"]), sharey=True)
    all_lo = min(min(data[p["key"]]["ci95"]) for p in panels)
    all_hi = max(max(data[p["key"]]["ci95"]) for p in panels)
    all_lo = min(all_lo, data["matched"]["ci95"][0])
    all_hi = max(all_hi, data["matched"]["ci95"][1])
    pad = float(f3["x_pad"])
    color = pal["confirmatory"]
    halo = {"boxstyle": "round,pad=0.15", "fc": style["surface"], "ec": "none"}
    reference = data["folded"]["point"]
    for ax, panel in zip(axes, panels):
        item = data[panel["key"]]
        ax.axvline(0, color=style["zero_line"], lw=0.8, zorder=1)
        if panel["key"] == "yes_rate":
            ax.axvline(reference, color=style["axis"], lw=0.8, ls=(0, (3, 2)), zorder=1)
            ax.text(reference, -0.5, f"dashed: confidence effect\n({reference:+.2f}), not excluded", ha="center", va="top", fontsize=style["annotation_size"] - 1.0, color=style["text_secondary"], bbox=halo, zorder=4)
        ax.plot(item["ci95"], [0, 0], color=color, lw=style["line_width"], solid_capstyle="butt", zorder=2)
        ax.plot(item["point"], 0, marker="o", ms=style["marker_size"], color=color, mec=style["surface"], mew=style["surface_ring"], ls="none", zorder=3)
        ax.text(item["point"], 0.13, f"{item['point']:+.2f}", ha="center", va="bottom", fontsize=style["annotation_size"], color=style["text_primary"], bbox=halo, zorder=4)
        ax.text(item["point"], -0.13, f"[{item['ci95'][0]:+.2f}, {item['ci95'][1]:+.2f}]", ha="center", va="top", fontsize=style["annotation_size"] - 0.5, color=style["text_secondary"], bbox=halo, zorder=4)
        if panel["key"] == "standardized":
            m = data["matched"]
            grey = pal["roles"]["other"]
            ax.plot(m["ci95"], [-0.5, -0.5], color=grey, lw=1.2, zorder=2)
            ax.plot(m["point"], -0.5, marker="o", ms=style["marker_size"], mfc=style["surface"], mec=grey, mew=1.4, ls="none", zorder=3)
            ax.text(m["ci95"][1] + 0.12, -0.5, f"{m['point']:+.2f} [{m['ci95'][0]:+.2f}, {m['ci95'][1]:+.2f}]\n{f3['matched_label']}", ha="left", va="center", fontsize=style["annotation_size"] - 1.5, color=style["text_secondary"], bbox=halo, zorder=4)
        if f3["shared_x_range"]:
            ax.set_xlim(all_lo - pad, all_hi + pad)
        ax.set_ylim(-0.85, 0.45)
        ax.set_yticks([])
        ax.set_xlabel(f"{panel['label']}\n({panel['unit']})")
        ax.grid(axis="x", color=style["grid"], lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        _tidy(ax, style)
        ax.spines["left"].set_visible(False)
    fig.suptitle(_wrap(f3["title"], 100), x=0.01, ha="left", fontsize=style["title_size"])
    caption = f3["caption_template"].format(reps=data["reps"])
    fig.text(0.01, -0.06, _wrap(caption, 135), ha="left", va="top", fontsize=style["annotation_size"], color=style["text_secondary"])
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    paths = _save(fig, out_dir, f3["file"], cfg)
    return paths, data, caption


FIGURES = {"fig1": figure1, "fig2": figure2, "fig3": figure3}
