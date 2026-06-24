# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Co-adaptive FBCSP-LDA engine (shared across ALS / SHU / Stieger).
================================================================
Pseudo-online, two-class (L vs R) motor imagery decoding.

The idea is a prequential (test-then-train) loop on a chronological
trial stream:

  - Train an initial FBCSP-LDA model on the first INIT_PER_CLASS trials
    of each class. This same model is also kept frozen as a baseline.
  - Stream the remaining trials one at a time. For every trial we first
    PREDICT (with the adaptive model and, separately, the frozen one),
    then buffer the trial with its true label.
  - Once the buffer holds WIN_PER_CLASS trials of each class, that block
    is closed: we log the block accuracy of both models, then refit the
    adaptive model's spatial filters AND classifier on that latest block
    only (sliding window, no memory of older blocks). The frozen model
    never changes.

So what adapts is the CSP filter bank + the LDA, jointly. The frozen
model isolates how much the adaptation itself buys us.

The per-dataset runners only handle loading and build, for each subject,
a chronological list of trials [n_channels x n_samples] plus labels in
{1,2}. Everything below is dataset-agnostic.
"""

import os
import numpy as np
import pandas as pd
import scipy.signal as sig
from scipy.stats import zscore

import mne
mne.set_log_level("ERROR")
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG (defaults; runners may override via Config)
# ============================================================
class Config:
    bands          = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]
    n_csp          = 4      # CSP components kept per band
    n_select       = 5      # FBCSP features after MI selection
    init_per_class = 30     # trials/class for the initial (and frozen) model
    win_per_class  = 15     # trials/class per sliding update block
    save_dpi       = 600
    save_fmt       = "png"


PALETTE = {
    "adaptive": "#2166ac",
    "frozen":   "#b2182b",
    "jump":     "#d73027",
    "ambiguous": "#4575b4",
    "drift":    "#1b7837",
    "neutral":  "#555555",
}

# match the look of the non-stationarity scripts
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif"],
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   7,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "grid.linewidth":    0.5,
    "grid.alpha":        0.35,
    "lines.linewidth":   1.2,
})


def savefig(fig, path, dpi=Config.save_dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# FBCSP  (filter bank CSP + mutual-information feature selection)
# ============================================================
class FBCSP:
    """Bank of band-limited CSPs followed by top-k MI feature selection.

    Trials go in as raw (broadband) [n_trials x n_ch x n_samples]; each
    band is filtered internally so a single model owns all its filters.
    Ledoit-Wolf shrinkage keeps the covariance estimate sane on the small
    (~30 trial) windows we refit on.
    """

    def __init__(self, bands, fs, n_csp=4, n_select=5):
        self.bands     = bands
        self.fs        = fs
        self.n_csp     = n_csp
        self.n_select  = n_select
        nyq            = 0.5 * fs
        self._sos      = [sig.butter(4, [lo / nyq, hi / nyq],
                                     btype="bandpass", output="sos")
                          for lo, hi in bands]

    def _band(self, X, b):
        # X: [n_trials x n_ch x n_samples], zero-phase filter along time
        return sig.sosfiltfilt(self._sos[b], X, axis=2)

    def fit(self, X, y):
        self.csps = []
        feats     = []
        for b in range(len(self.bands)):
            csp = CSP(n_components=self.n_csp, reg="ledoit_wolf",
                      log=True, norm_trace=True,
                      component_order="mutual_info")
            Xb = self._band(X, b)
            csp.fit(Xb, y)
            self.csps.append(csp)
            feats.append(csp.transform(Xb))
        F = np.hstack(feats)                       # [n x n_bands*n_csp]

        # MIBIF-style selection: keep the n_select most informative columns
        mi          = mutual_info_classif(F, y, random_state=0)
        self.sel    = np.sort(np.argsort(mi)[::-1][:self.n_select])
        return self

    def transform(self, X):
        feats = [self.csps[b].transform(self._band(X, b))
                 for b in range(len(self.bands))]
        return np.hstack(feats)[:, self.sel]


class FBCSPLDA:
    """FBCSP features -> standardise -> shrinkage LDA."""

    def __init__(self, bands, fs, n_csp=4, n_select=5):
        self.fbcsp  = FBCSP(bands, fs, n_csp, n_select)
        self.scaler = StandardScaler()
        self.lda    = LinearDiscriminantAnalysis(solver="lsqr",
                                                 shrinkage="auto")

    def fit(self, X, y):
        F = self.fbcsp.fit(X, y).transform(X)
        self.lda.fit(self.scaler.fit_transform(F), y)
        return self

    def predict(self, X):
        if X.ndim == 2:                            # single trial -> batch of 1
            X = X[None]
        F = self.fbcsp.transform(X)
        return self.lda.predict(self.scaler.transform(F))


# ============================================================
# ONLINE CO-ADAPTATION
# ============================================================
def _take_balanced(buf_X, buf_y, n_per_class):
    """Most-recent n_per_class trials of each class from a block buffer."""
    X = np.stack(buf_X)
    y = np.asarray(buf_y)
    keep = []
    for cls in (1, 2):
        idx = np.where(y == cls)[0]
        keep.extend(idx[-n_per_class:])
    keep = np.sort(keep)
    return X[keep], y[keep]


def run_subject(trials, labels, fs, cfg=Config, session_bounds=None):
    """Run the pseudo-online loop for one subject.

    trials  : list of [n_ch x n_samples] arrays in chronological order
    labels  : matching labels in {1, 2}
    returns : (blocks_df, summary_dict) or None if the subject is too short
    """
    labels = np.asarray(labels, dtype=int)
    n_init = cfg.init_per_class

    # --- carve off the initial training set (first n_init of each class) ---
    seen = {1: 0, 2: 0}
    cut  = None
    for i, lab in enumerate(labels):
        seen[lab] += 1
        if seen[1] >= n_init and seen[2] >= n_init:
            cut = i + 1
            break
    if cut is None:
        return None                                # not enough to even start

    init_idx = [i for i in range(cut)]
    Xi = np.stack([trials[i] for i in init_idx])
    yi = labels[init_idx]

    adaptive = FBCSPLDA(cfg.bands, fs, cfg.n_csp, cfg.n_select).fit(Xi, yi)
    frozen   = FBCSPLDA(cfg.bands, fs, cfg.n_csp, cfg.n_select).fit(Xi, yi)

    # --- stream the rest, test-then-train in balanced blocks ---
    bounds  = set(session_bounds or [])
    blocks  = []
    buf_X, buf_y, buf_a, buf_f, buf_i = [], [], [], [], []
    blk_start = cut

    def close_block(retrain):
        nonlocal adaptive, blk_start, buf_X, buf_y, buf_a, buf_f, buf_i
        y    = np.asarray(buf_y)
        a_ok = (np.asarray(buf_a) == y).mean()
        f_ok = (np.asarray(buf_f) == y).mean()
        # is a session boundary crossed inside this block?
        sess = any(b in bounds for b in range(blk_start, buf_i[-1] + 1))
        blocks.append({
            "block":        len(blocks) + 1,
            "trial_start":  blk_start,
            "trial_end":    buf_i[-1],
            "n":            len(buf_y),
            "n_L":          int((y == 1).sum()),
            "n_R":          int((y == 2).sum()),
            "new_session":  sess,
            "acc_adaptive": a_ok,
            "acc_frozen":   f_ok,
        })
        if retrain:
            Xw, yw   = _take_balanced(buf_X, buf_y, cfg.win_per_class)
            adaptive = FBCSPLDA(cfg.bands, fs, cfg.n_csp,
                                cfg.n_select).fit(Xw, yw)
        blk_start = buf_i[-1] + 1
        buf_X, buf_y, buf_a, buf_f, buf_i = [], [], [], [], []

    for i in range(cut, len(trials)):
        x   = trials[i]
        lab = labels[i]
        buf_a.append(int(adaptive.predict(x)[0]))
        buf_f.append(int(frozen.predict(x)[0]))
        buf_X.append(x)
        buf_y.append(lab)
        buf_i.append(i)

        n_L = sum(1 for v in buf_y if v == 1)
        n_R = sum(1 for v in buf_y if v == 2)
        if n_L >= cfg.win_per_class and n_R >= cfg.win_per_class:
            close_block(retrain=True)

    # trailing partial block: score it but don't bother retraining
    if len(buf_y) >= cfg.win_per_class:
        close_block(retrain=False)

    if not blocks:
        return None

    blocks_df = pd.DataFrame(blocks)
    summary = {
        "n_blocks":      len(blocks_df),
        "init_trials":   cut,
        "n_stream":      len(trials) - cut,
        "acc_adaptive":  blocks_df["acc_adaptive"].mean(),
        "acc_frozen":    blocks_df["acc_frozen"].mean(),
        "acc_first":     blocks_df["acc_adaptive"].iloc[0],
        "acc_last":      blocks_df["acc_adaptive"].iloc[-1],
    }
    summary["recovery"] = summary["acc_adaptive"] - summary["acc_frozen"]
    return blocks_df, summary


# ============================================================
# STREAM PREP
# ============================================================
def zscore_trials(trials):
    """Per-trial, per-channel z-score over time (matches the CSP scripts)."""
    return [zscore(t, axis=1) for t in trials]


def standardize_lengths(trials, labels, bounds, thresh=0.5):
    """Pad/trim trials to the subject's modal length; drop very short ones.

    Trial duration is a fixed epoch parameter, so using the global modal
    length here is not distributional leakage. Trials shorter than `thresh`
    of the modal length are discarded (matches the CSP scripts' 50% rule).
    Session boundaries are remapped to the surviving trials.
    """
    labels  = np.asarray(labels, dtype=int)
    lengths = np.array([t.shape[1] for t in trials])
    vals, counts = np.unique(lengths, return_counts=True)
    target = int(vals[np.argmax(counts)])

    out_t, out_y, kept = [], [], []
    for i, (t, y) in enumerate(zip(trials, labels)):
        T = t.shape[1]
        if T < thresh * target:
            continue
        if T > target:
            t = t[:, :target]
        elif T < target:
            t = np.concatenate([t, np.zeros((t.shape[0], target - T))], axis=1)
        out_t.append(t)
        out_y.append(y)
        kept.append(i)

    # remap each boundary to how many surviving trials precede it
    kept = np.asarray(kept)
    new_bounds = sorted({int(np.searchsorted(kept, b)) for b in (bounds or [])})
    return out_t, np.asarray(out_y, dtype=int), new_bounds


# ============================================================
# FIGURES
# ============================================================
def plot_subject(dataset, sid, regime, blocks_df, fig_dir, cfg=Config):
    """Adaptive vs frozen block accuracy across the trial stream."""
    x = blocks_df["trial_end"].values
    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    ax.plot(x, blocks_df["acc_adaptive"], "o-", color=PALETTE["adaptive"],
            markersize=4, label="Co-adaptive")
    ax.plot(x, blocks_df["acc_frozen"], "s--", color=PALETTE["frozen"],
            markersize=4, label="Frozen")
    ax.axhline(0.5, color="gray", ls=":", lw=0.8, label="Chance")

    # mark sessions where they exist
    first = True
    for _, row in blocks_df[blocks_df["new_session"]].iterrows():
        ax.axvline(row["trial_start"], color="0.7", lw=0.7, ls="-",
                   label="New session" if first else None)
        first = False

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Trial index")
    ax.set_ylabel("Block accuracy")
    ax.set_title(f"{sid}  ({regime})")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.grid(True)
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, f"{dataset}_{sid}_online.{cfg.save_fmt}"),
            cfg.save_dpi)


def plot_cohort(dataset, all_blocks, regimes, fig_dir, cfg=Config):
    """Mean +/- SEM block accuracy vs block index, split by regime.

    Subjects have different numbers of blocks, so we align on block index
    and average where subjects overlap.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4))

    by_regime = {}
    for sid, df in all_blocks.items():
        by_regime.setdefault(regimes.get(sid, "neutral"), []).append(df)

    for regime, dfs in sorted(by_regime.items()):
        col = PALETTE.get(regime, PALETTE["neutral"])
        for key, ls, mark, tag in [("acc_adaptive", "-", "o", "adaptive"),
                                   ("acc_frozen",  "--", "s", "frozen")]:
            max_b = max(len(d) for d in dfs)
            mat   = np.full((len(dfs), max_b), np.nan)
            for r, d in enumerate(dfs):
                mat[r, :len(d)] = d[key].values
            xs   = np.arange(1, max_b + 1)
            mean = np.nanmean(mat, axis=0)
            n    = np.sum(~np.isnan(mat), axis=0)
            sem  = np.nanstd(mat, axis=0) / np.sqrt(np.maximum(n, 1))
            alpha = 1.0 if tag == "adaptive" else 0.55
            ax.plot(xs, mean, ls, marker=mark, color=col, markersize=3,
                    alpha=alpha,
                    label=f"{regime} ({tag}, n={len(dfs)})")
            ax.fill_between(xs, mean - sem, mean + sem, color=col, alpha=0.12)

    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Update block")
    ax.set_ylabel("Block accuracy")
    ax.set_title(f"{dataset}: co-adaptive vs frozen")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    ax.grid(True)
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, f"{dataset}_cohort_online.{cfg.save_fmt}"),
            cfg.save_dpi)


def plot_recovery(dataset, summary_df, fig_dir, cfg=Config):
    """Per-subject overall adaptive vs frozen, paired, coloured by regime."""
    df = summary_df.sort_values("recovery", ascending=False).reset_index(drop=True)
    x  = np.arange(len(df))
    w  = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4),
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    ax1.bar(x - w / 2, df["acc_adaptive"], w, color=PALETTE["adaptive"],
            label="Co-adaptive")
    ax1.bar(x + w / 2, df["acc_frozen"], w, color=PALETTE["frozen"],
            label="Frozen")
    ax1.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["subject"], rotation=45, ha="right")
    ax1.set_ylabel("Mean block accuracy")
    ax1.set_ylim(0, 1.0)
    ax1.set_title(f"{dataset}: adaptive vs frozen per subject")
    ax1.legend(frameon=False)
    ax1.grid(axis="y")

    cols = [PALETTE.get(r, PALETTE["neutral"]) for r in df["regime"]]
    ax2.bar(x, df["recovery"], color=cols)
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["subject"], rotation=45, ha="right")
    ax2.set_ylabel("Recovery  (adaptive - frozen)")
    ax2.set_title("Adaptation benefit")
    ax2.grid(axis="y")
    # regime legend
    seen = {}
    for r, c in zip(df["regime"], cols):
        seen[r] = c
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in seen.values()]
    ax2.legend(handles, seen.keys(), frameon=False, title="Regime")

    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, f"{dataset}_recovery.{cfg.save_fmt}"),
            cfg.save_dpi)


# ============================================================
# TABLES + DRIVER
# ============================================================
def run_dataset(dataset, subject_streams, regimes, fig_dir, fs, cfg=Config):
    """Run every subject, write tables + figures. Returns the summary frame.

    subject_streams : {sid: (trials, labels, session_bounds)}
    regimes         : {sid: "jump"/"ambiguous"/...}
    """
    os.makedirs(fig_dir, exist_ok=True)
    all_blocks  = {}
    summary_rows = []

    for sid in subject_streams:
        trials, labels, bounds = subject_streams[sid]
        trials, labels, bounds = standardize_lengths(trials, labels, bounds)
        trials = zscore_trials(trials)
        out = run_subject(trials, labels, fs, cfg, bounds)
        if out is None:
            print(f"[SKIP] {sid}: too few trials")
            continue
        blocks_df, summary = out
        all_blocks[sid] = blocks_df

        blocks_df.insert(0, "subject", sid)
        summary_rows.append({"subject": sid,
                             "regime": regimes.get(sid, "?"),
                             **summary})

        plot_subject(dataset, sid, regimes.get(sid, "?"), blocks_df, fig_dir, cfg)
        print(f"[{sid}] {regimes.get(sid,'?'):>9} | "
              f"adaptive={summary['acc_adaptive']:.3f} "
              f"frozen={summary['acc_frozen']:.3f} "
              f"recovery={summary['recovery']:+.3f} "
              f"({summary['n_blocks']} blocks)")

    if not summary_rows:
        print("[WARN] no subjects produced results")
        return None

    summary_df = pd.DataFrame(summary_rows)

    # long per-block table
    long_df = pd.concat(all_blocks.values(), ignore_index=True)
    long_df.to_csv(os.path.join(fig_dir, f"{dataset}_blocks.csv"), index=False)

    # subjects x block accuracy matrix (adaptive) - easy to drop in a thesis
    wide = {sid: df.set_index("block")["acc_adaptive"]
            for sid, df in all_blocks.items()}
    pd.DataFrame(wide).to_csv(
        os.path.join(fig_dir, f"{dataset}_accuracy_over_time.csv"))

    summary_df.to_csv(os.path.join(fig_dir, f"{dataset}_summary.csv"),
                      index=False)

    plot_cohort(dataset, all_blocks, regimes, fig_dir, cfg)
    plot_recovery(dataset, summary_df, fig_dir, cfg)

    print(f"\n[SUMMARY] {dataset}")
    print(summary_df.round(3).to_string(index=False))
    print(f"[DONE] outputs in {fig_dir}")
    return summary_df
