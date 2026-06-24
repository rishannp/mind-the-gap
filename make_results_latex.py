# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Emit LaTeX for the co-adaptation results: one subfigure-grid per dataset
(block-accuracy plots) and one combined summary table across all datasets.

Reads the per-dataset *_summary.csv files written by coadapt_core. Datasets
that haven't been run yet (e.g. Stieger before the cluster run) are skipped,
so just re-run this once Stieger_summary.csv is copied back to get the full
table/figures.
"""

import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))

# dataset -> (summary csv, caption stub, n figure columns)
DATASETS = {
    "ALS":     os.path.join(ROOT, "ALS",     "als_coadapt_figures",     "ALS_summary.csv"),
    "SHU":     os.path.join(ROOT, "SHU",     "shu_coadapt_figures",     "SHU_summary.csv"),
    "Stieger": os.path.join(ROOT, "Stieger", "stieger_coadapt_figures", "Stieger_summary.csv"),
}

# prepended to every \includegraphics path - set to your thesis figures dir
FIG_PREFIX = ""

CAPTIONS = {
    "ALS": ("Pseudo-online block accuracy for all eight ALS subjects under the "
            "co-adaptive FBCSP--LDA model (blue) and the static frozen baseline "
            "(red), relative to chance (0.5, dotted). Each marker is the accuracy "
            "over one 30-trial update block; the co-adaptive model re-estimates "
            "its spatial filters and LDA on the most recent block only. Subject "
            "regime (jump/ambiguous) from the non-stationarity analysis is given "
            "in each panel title."),
    "SHU": ("Pseudo-online block accuracy for the five SHU subjects with the "
            "largest jump-regime accuracy collapse, under the co-adaptive "
            "FBCSP--LDA model (blue) and the static frozen baseline (red). Grey "
            "vertical lines mark recording-session boundaries; the dotted line is "
            "chance. Each marker is one 30-trial update block."),
    "Stieger": ("Pseudo-online block accuracy for the five Stieger subjects with "
                "the largest jump-regime accuracy collapse, under the co-adaptive "
                "FBCSP--LDA model (blue) and the static frozen baseline (red). Grey "
                "vertical lines mark session boundaries; the dotted line is chance."),
}


def _label(s):
    return s.lower().replace("_", "").replace("-", "")


def _num(subject):
    """Trailing integer in a subject id, for natural sorting (S5 < S39)."""
    digits = "".join(ch for ch in str(subject) if ch.isdigit())
    return int(digits) if digits else 0


def figure_block(dataset, df, ncols=2):
    sub_w = round(0.97 / ncols, 2)
    lines = ["\\begin{figure}[htbp]", "  \\centering"]
    rows = list(df.itertuples())
    for i, r in enumerate(rows):
        fname = f"{FIG_PREFIX}{dataset}_{r.subject}_online.png"
        lines += [
            f"  \\begin{{subfigure}}[t]{{{sub_w}\\textwidth}}",
            "    \\centering",
            f"    \\includegraphics[width=\\linewidth]{{{fname}}}",
            f"    \\caption{{{r.subject} ({r.regime})}}",
            f"    \\label{{fig:online-{_label(dataset)}-{_label(r.subject)}}}",
            "  \\end{subfigure}",
        ]
        if (i + 1) % ncols == 0 and (i + 1) < len(rows):
            lines.append("  \\par\\medskip")
        elif (i + 1) < len(rows):
            lines.append("  \\hfill")
    lines += [
        f"  \\caption{{{CAPTIONS[dataset]}}}",
        f"  \\label{{fig:online-{_label(dataset)}}}",
        "\\end{figure}",
        "",
    ]
    return "\n".join(lines)


def _mean_row(label, sub, bold=True):
    f, a = sub["acc_frozen"].mean(), sub["acc_adaptive"].mean()
    rec = a - f
    name = f"\\textbf{{{label}}}" if bold else label
    cells = (f"{name} & & & \\textbf{{{f:.3f}}} & \\textbf{{{a:.3f}}} "
             f"& \\textbf{{{rec:+.3f}}}")
    return cells + " \\\\"


def table_block(frames):
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Mean pseudo-online block accuracy of the static (frozen) "
        "versus co-adaptive FBCSP--LDA model, and the recovery (co-adaptive "
        "$-$ frozen), per subject and averaged by dataset and regime. Positive "
        "recovery indicates the co-adaptation improved on the static model; "
        "values near zero or negative indicate the distributional shift was not "
        "recovered. Subjects were selected as described in "
        "Section~\\ref{sec:subject-selection}.}",
        "  \\label{tab:coadapt-summary}",
        "  \\begin{tabular}{lllccc}",
        "    \\toprule",
        "    Dataset & Subject & Regime & Frozen & Co-adaptive & Recovery \\\\",
        "    \\midrule",
    ]
    all_rows = []
    for dataset, df in frames.items():
        df = df.assign(_k=df["subject"].map(_num))
        df = df.sort_values(["regime", "_k"]).drop(columns="_k").reset_index(drop=True)
        all_rows.append(df.assign(dataset=dataset))
        for j, r in enumerate(df.itertuples()):
            ds = dataset if j == 0 else ""
            lines.append(
                f"    {ds} & {r.subject} & {r.regime} & {r.acc_frozen:.3f} "
                f"& {r.acc_adaptive:.3f} & {r.recovery:+.3f} \\\\")
        # regime subtotals if the dataset mixes regimes, else just dataset mean
        lines.append("    \\cmidrule(l){2-6}")
        if df["regime"].nunique() > 1:
            for reg, sub in df.groupby("regime"):
                lines.append("    " + _mean_row(f"{dataset} ({reg})", sub))
        lines.append("    " + _mean_row(f"{dataset} (all)", df))
        lines.append("    \\midrule")

    pooled = pd.concat(all_rows, ignore_index=True)
    lines.append("    " + _mean_row("Overall", pooled))
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    frames, figures = {}, []
    for dataset, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"% [skip] {dataset}: {os.path.basename(path)} not found")
            continue
        df = pd.read_csv(path)
        frames[dataset] = df
        figures.append(figure_block(dataset, df))

    out = "\n".join(figures) + "\n" + table_block(frames)
    with open(os.path.join(ROOT, "coadapt_results.tex"), "w") as fh:
        fh.write(out)
    print(out)
    print(f"\n% written to {os.path.join(ROOT, 'coadapt_results.tex')}")
