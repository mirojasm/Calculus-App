"""
Collaborative advantage and CPS richness metrics.

collaborative_advantage(solo, collab) → scalar
cps_necessity(jigsaw, unrestricted)   → scalar  (does split FORCE more CPS?)
phase_advantage(conversations)        → per-phase breakdown
build_results_df(records)             → pandas DataFrame for analysis
"""
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy import stats


# ── result record ─────────────────────────────────────────────────────────────

@dataclass
class ExperimentRecord:
    """One row in the results table — one conversation fully scored."""
    problem_id:   str
    subject:      str
    level:        int          # 1–5
    openness:     str          # "open" | "closed"
    condition:    str          # solo / unrestricted_pair / jigsaw_2 / jigsaw_3 / jigsaw_4
    n_agents:     int
    correct:      bool         # did the final answer match ground truth?
    consensus:    bool
    total_turns:  int
    # PISA
    pisa_global:  float
    pisa_entropy: float
    pisa_process_A: float
    pisa_process_B: float
    pisa_process_C: float
    pisa_process_D: float
    pisa_comp_1:  float
    pisa_comp_2:  float
    pisa_comp_3:  float
    # ATC21S
    atc_global:   float
    atc_social:   float
    atc_cognitive: float
    atc_PC:  float
    atc_C:   float
    atc_Co:  float
    atc_CR:  float
    atc_SR:  float
    split_pattern: str = ""


def record_to_dict(r: ExperimentRecord) -> dict:
    return r.__dict__


def build_results_df(records: List[ExperimentRecord]) -> pd.DataFrame:
    return pd.DataFrame([record_to_dict(r) for r in records])


# ── answer comparison ─────────────────────────────────────────────────────────

def _normalise_answer(ans: Optional[str]) -> str:
    if ans is None:
        return ""
    import re
    # Strip LaTeX wrappers, punctuation, units, whitespace — keep digits/operators
    ans = re.sub(r"\\(text|mathrm|mathbf|operatorname)\{[^}]*\}", "", ans)
    ans = re.sub(r"\\[a-zA-Z]+", "", ans)       # remaining latex commands
    ans = re.sub(r"[{}\[\]()$,;]", "", ans)
    ans = re.sub(r"\s+", "", ans).lower()
    ans = ans.rstrip(".").strip()
    return ans


def is_correct(predicted: Optional[str], ground_truth: Optional[str]) -> bool:
    """Fuzzy normalised string match for math answers."""
    if not predicted or not ground_truth:
        return False
    pn = _normalise_answer(predicted)
    gn = _normalise_answer(ground_truth)
    if pn == gn:
        return True
    # Also check if ground truth appears as substring (handles "20 cm" vs "20")
    return gn in pn or pn in gn


# ── collaborative advantage ───────────────────────────────────────────────────

def collaborative_advantage(
    df: pd.DataFrame,
    group_by: List[str] = ["level"],
    collab_condition: str = "jigsaw_2",
    solo_condition:   str = "solo",
) -> pd.DataFrame:
    """
    Computes advantage = P(correct | collab) - P(correct | solo)
    grouped by the specified columns.
    Adds n_solo, n_collab, p_value (Mann-Whitney U), and cohens_d.
    """
    solo   = df[df["condition"] == solo_condition]
    collab = df[df["condition"] == collab_condition]

    solo_rate   = solo.groupby(group_by)["correct"].mean().rename("solo_success")
    collab_rate = collab.groupby(group_by)["correct"].mean().rename("collab_success")
    solo_n      = solo.groupby(group_by)["correct"].count().rename("n_solo")
    collab_n    = collab.groupby(group_by)["correct"].count().rename("n_collab")

    adv = pd.concat([solo_rate, collab_rate, solo_n, collab_n], axis=1)
    adv["collaborative_advantage"] = adv["collab_success"] - adv["solo_success"]

    # Per-group Mann-Whitney U test and Cohen's d on binary correct column
    p_values = []
    cohens_ds = []
    for key, _ in adv.iterrows():
        key_tuple = key if isinstance(key, tuple) else (key,)
        mask_s = pd.Series([True] * len(solo))
        mask_c = pd.Series([True] * len(collab))
        for col, val in zip(group_by, key_tuple):
            mask_s = mask_s & (solo[col].values == val)
            mask_c = mask_c & (collab[col].values == val)
        s_vals = solo["correct"].values[mask_s.values].astype(float)
        c_vals = collab["correct"].values[mask_c.values].astype(float)
        if len(s_vals) >= 2 and len(c_vals) >= 2:
            _, p = stats.mannwhitneyu(s_vals, c_vals, alternative="two-sided")
            pooled_std = np.sqrt((np.std(s_vals, ddof=1)**2 + np.std(c_vals, ddof=1)**2) / 2)
            d = (np.mean(c_vals) - np.mean(s_vals)) / pooled_std if pooled_std > 0 else np.nan
        else:
            p, d = np.nan, np.nan
        p_values.append(p)
        cohens_ds.append(d)

    adv["p_value"]  = p_values
    adv["cohens_d"] = cohens_ds
    return adv.reset_index()


def cps_necessity(
    df: pd.DataFrame,
    jigsaw_condition:       str = "jigsaw_2",
    unrestricted_condition: str = "unrestricted_pair",
    metric: str = "pisa_global",
) -> pd.DataFrame:
    """
    CPS necessity effect = mean(CPS | jigsaw) - mean(CPS | unrestricted)
    Tests whether information splitting FORCES richer CPS behavior.
    Adds n_jigsaw, n_unrestricted, p_value (Mann-Whitney U), cohens_d.
    """
    group_cols = ["level", "subject"]
    jig_df = df[df["condition"] == jigsaw_condition][group_cols + [metric]]
    unr_df = df[df["condition"] == unrestricted_condition][group_cols + [metric]]

    jig_mean = jig_df.groupby(group_cols)[metric].mean().rename("jigsaw_cps")
    unr_mean = unr_df.groupby(group_cols)[metric].mean().rename("unrestricted_cps")
    jig_n    = jig_df.groupby(group_cols)[metric].count().rename("n_jigsaw")
    unr_n    = unr_df.groupby(group_cols)[metric].count().rename("n_unrestricted")

    result = pd.concat([jig_mean, unr_mean, jig_n, unr_n], axis=1).dropna(subset=["jigsaw_cps", "unrestricted_cps"])
    result["necessity_effect"] = result["jigsaw_cps"] - result["unrestricted_cps"]

    # Per subject×level Mann-Whitney U test and Cohen's d
    p_values = []
    cohens_ds = []
    for (level, subject), _ in result.iterrows():
        j_vals = jig_df[(jig_df["level"] == level) & (jig_df["subject"] == subject)][metric].values
        u_vals = unr_df[(unr_df["level"] == level) & (unr_df["subject"] == subject)][metric].values
        if len(j_vals) >= 2 and len(u_vals) >= 2:
            _, p = stats.mannwhitneyu(j_vals, u_vals, alternative="two-sided")
            pooled_std = np.sqrt((np.std(j_vals, ddof=1)**2 + np.std(u_vals, ddof=1)**2) / 2)
            d = (np.mean(j_vals) - np.mean(u_vals)) / pooled_std if pooled_std > 0 else np.nan
        else:
            p, d = np.nan, np.nan
        p_values.append(p)
        cohens_ds.append(d)

    result["p_value"]  = p_values
    result["cohens_d"] = cohens_ds
    return result.reset_index()


def group_size_effect(
    df: pd.DataFrame,
    metric: str = "pisa_global",
) -> pd.DataFrame:
    """Mean metric by N and complexity level."""
    jigsaw = df[df["condition"].str.startswith("jigsaw")]
    jigsaw = jigsaw.copy()
    jigsaw["n"] = jigsaw["n_agents"]
    return (
        jigsaw.groupby(["n", "level"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def phase_advantage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compares PISA phase shares (A,B,C,D) between jigsaw_2 and solo.
    Shows which problem-solving phase benefits most from collaboration.
    Adds p_value (Wilcoxon signed-rank on per-conversation phase share values).
    """
    conditions = ["solo", "jigsaw_2"]
    cols = ["pisa_process_A", "pisa_process_B", "pisa_process_C", "pisa_process_D"]
    sub = df[df["condition"].isin(conditions)]

    means = (
        sub.groupby("condition")[cols]
        .mean()
        .T.rename(columns={"solo": "solo_mean", "jigsaw_2": "jigsaw_mean"})
        .assign(phase_advantage=lambda x: x["jigsaw_mean"] - x["solo_mean"])
    )

    # Wilcoxon signed-rank on raw per-conversation process share values
    solo_raw   = df[df["condition"] == "solo"]
    collab_raw = df[df["condition"] == "jigsaw_2"]
    p_values = []
    for col in cols:
        s_vals = solo_raw[col].dropna().values
        c_vals = collab_raw[col].dropna().values
        # Pair up by minimum length (independent samples — use Wilcoxon rank-sum / Mann-Whitney)
        n = min(len(s_vals), len(c_vals))
        if n >= 2:
            try:
                _, p = stats.wilcoxon(c_vals[:n] - s_vals[:n])
            except ValueError:
                p = np.nan
        else:
            p = np.nan
        p_values.append(p)

    means["p_value"] = p_values
    return means


def pisa_vs_atc_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson r between PISA global index and each ATC21S dimension."""
    atc_dims = ["atc_PC", "atc_C", "atc_Co", "atc_CR", "atc_SR", "atc_global"]
    rows = []
    for dim in atc_dims:
        r, p = stats.pearsonr(df["pisa_global"].dropna(), df[dim].dropna())
        rows.append({"atc_dimension": dim, "pearson_r": r, "p_value": p})
    return pd.DataFrame(rows)


def problem_type_summary(
    df: pd.DataFrame,
    condition: str = "jigsaw_2",
) -> pd.DataFrame:
    """CPS richness and success rate by subject for a given condition."""
    sub = df[df["condition"] == condition]
    return (
        sub.groupby("subject")
        .agg(
            success_rate=("correct", "mean"),
            pisa_global=("pisa_global", "mean"),
            atc_global=("atc_global", "mean"),
            entropy=("pisa_entropy", "mean"),
            n=("problem_id", "count"),
        )
        .reset_index()
        .sort_values("pisa_global", ascending=False)
    )


def openness_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Open vs closed problem CPS richness and success by condition."""
    return (
        df.groupby(["openness", "condition"])
        .agg(
            success_rate=("correct", "mean"),
            pisa_global=("pisa_global", "mean"),
            pisa_entropy=("pisa_entropy", "mean"),
            atc_global=("atc_global", "mean"),
            n=("problem_id", "count"),
        )
        .reset_index()
    )


def competence_advantage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compares PISA competence shares (1, 2, 3) across all conditions.

    Competence 1: shared knowledge / communication of meaning
    Competence 2: mathematical action / cognitive execution
    Competence 3: organization, coordination, delegation

    Shows how the collaborative structure shifts which social competency dominates.
    """
    conditions = ["solo", "unrestricted_pair", "jigsaw_2", "jigsaw_3", "jigsaw_4"]
    cols = ["pisa_comp_1", "pisa_comp_2", "pisa_comp_3"]
    sub = df[df["condition"].isin(conditions)]
    result = (
        sub.groupby("condition")[cols]
        .mean()
        .reindex(conditions)
        .dropna(how="all")
        .rename(columns={
            "pisa_comp_1": "comp1_shared_knowledge",
            "pisa_comp_2": "comp2_math_action",
            "pisa_comp_3": "comp3_coordination",
        })
    )
    result["solo_baseline_comp2"] = result.loc["solo", "comp2_math_action"] if "solo" in result.index else np.nan
    result["comp2_shift"] = result["comp2_math_action"] - result["solo_baseline_comp2"]
    result = result.drop(columns=["solo_baseline_comp2"])
    return result


def competence_by_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean competence shares by condition × level.
    Reveals whether harder problems force more communication (comp1) or coordination (comp3).
    """
    conditions = ["solo", "jigsaw_2", "jigsaw_3", "jigsaw_4"]
    cols = ["pisa_comp_1", "pisa_comp_2", "pisa_comp_3"]
    sub = df[df["condition"].isin(conditions)]
    return (
        sub.groupby(["condition", "level"])[cols]
        .mean()
        .rename(columns={
            "pisa_comp_1": "comp1_shared_knowledge",
            "pisa_comp_2": "comp2_math_action",
            "pisa_comp_3": "comp3_coordination",
        })
        .reset_index()
    )


def split_pattern_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes CPS richness and outcome by jigsaw split pattern (SPLIT-A through SPLIT-G).
    Filters to jigsaw conditions only, groups by split_pattern, and returns per-pattern
    aggregates sorted by pisa_global descending.

    Requires that df contains a 'split_pattern' column (populated from split files).
    """
    jigsaw = df[df["condition"].str.startswith("jigsaw")].copy()

    if "split_pattern" not in jigsaw.columns or jigsaw["split_pattern"].isna().all():
        return pd.DataFrame(columns=[
            "split_pattern", "count", "mean_pisa_global", "mean_atc_global",
            "mean_pisa_comp_3", "mean_correct", "mean_total_turns",
        ])

    result = (
        jigsaw.groupby("split_pattern")
        .agg(
            count=("problem_id", "count"),
            mean_pisa_global=("pisa_global", "mean"),
            mean_atc_global=("atc_global", "mean"),
            mean_pisa_comp_3=("pisa_comp_3", "mean"),
            mean_correct=("correct", "mean"),
            mean_total_turns=("total_turns", "mean"),
        )
        .reset_index()
        .sort_values("mean_pisa_global", ascending=False)
    )
    return result
