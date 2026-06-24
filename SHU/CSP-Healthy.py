# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
SHU MI Dataset - Non-stationarity Characterisation
====================================================
- Load .mat files (sub-XXX_ses-XX format), bandpass filter (8-30 Hz), ASR
- Split each subject's trials causally into 5 pseudo-sessions
- Freeze CSP on Session 1, project all sessions
- Frozen SVM accuracy per session
- Symmetric KL divergence (within-session class separability)
- MMD: within-session separability + cross-session marginal drift
- Regime classification: gradual drift vs distributional jump
- Publication-quality figures
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import scipy.signal as sig
from scipy.io import loadmat
from scipy.stats import zscore
from numpy.linalg import inv

import mne
from mne.decoding import CSP
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from asrpy.asr import asr_calibrate, asr_process

# ============================================================
# CONFIG
# ============================================================
data_dir  = r'C:\Users\uceerjp\Desktop\PhD\Multi-Session Data\SHU Dataset\MatFiles'
fig_dir   = r'C:\Users\uceerjp\Desktop\PhD\THESIS\THESIS - Chapter 3\SHU\shu_nonstationarity_figures'
os.makedirs(fig_dir, exist_ok=True)

fs               = 250
band_low         = 8.0
band_high        = 30.0
asr_cutoff       = 20.0
USE_ASR          = False
n_csp_components = 2
N_FOLDS          = 5
N_PERMUTATIONS   = 1000
MIN_TRIALS_CLS   = 15
SAVE_DPI         = 600
SAVE_FMT         = "png"

# ── Publication style ──────────────────────────────────────
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

PALETTE = {
    "drift":   "#2166ac",
    "jump":    "#d73027",
    "neutral": "#555555",
}
SESSION_COLORS = ["#2166ac", "#4dac26", "#d01c8b", "#f1a340", "#762a83"]
CLASS_MARKERS  = {1: "s", 2: "o"}
CLASS_LABELS   = {1: "Left", 2: "Right"}

CH_NAMES = [
    'FP1','FP2','FZ','F3','F4','F7','F8','FC1','FC2','FC5',
    'FC6','CZ','C3','C4','T3','T4','A1','A2','CP1','CP2',
    'CP5','CP6','PZ','P3','P4','T5','T6','PO3','PO4','OZ',
    'O1','O2'
]
CH_COORDS = np.array([
    [ 80.79573128,  26.09631015,  -4.00404831],
    [ 80.79573128, -26.09631015,  -4.00404831],
    [ 60.73017777,   0.0,         59.47138394],
    [ 57.57616305,  48.14114469,  39.90508284],
    [ 57.57616305, -48.14114469,  39.90508284],
    [ 49.88651615,  68.41148946,  -7.49690713],
    [ 49.88728633, -68.41254564,  -7.48212953],
    [ 32.43878889,  32.32575332,  71.60845375],
    [ 32.43878889, -32.32575332,  71.60845375],
    [ 28.80808576,  76.2383868,   24.1413043 ],
    [ 28.80808576, -76.2383868,   24.1413043 ],
    [  5.20e-15,     0.0,         85.0        ],
    [  3.87e-15,    63.16731017,  56.87610154 ],
    [  3.87e-15,   -63.16731017,  56.87610154 ],
    [  5.17e-15,    84.5,         -8.85       ],
    [  5.17e-15,   -84.5,         -8.85       ],
    [  3.68e-15,    60.1,        -60.1        ],
    [  3.68e-15,   -60.1,        -60.1        ],
    [-32.38232042,  32.38232042,  71.60845375 ],
    [-32.38232042, -32.38232042,  71.60845375 ],
    [-29.2068723,   76.08650365,  24.1413043  ],
    [-29.2068723,  -76.08650365,  24.1413043  ],
    [-60.73017777,  -7.44e-15,    59.47138394 ],
    [-57.49205325,  48.24156068,  39.90508284 ],
    [-57.49205325, -48.24156068,  39.90508284 ],
    [-49.9,         68.4,         -7.49       ],
    [-49.9,        -68.4,         -7.49       ],
    [-76.40259649,  30.8686527,   20.8511278  ],
    [-76.40259649, -30.8686527,   20.8511278  ],
    [-84.9813581,   -1.04e-14,    -1.78010569 ],
    [-80.75006159,  26.23728548,  -4.00404831 ],
    [-80.75006159, -26.23728548,  -4.00404831 ],
])


# ============================================================
# HELPERS
# ============================================================
def _safe(s):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(s))

def savefig(fig, path):
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

def _sess_color(i):
    return SESSION_COLORS[i % len(SESSION_COLORS)]

def create_info():
    ch_pos  = {c: CH_COORDS[i] for i, c in enumerate(CH_NAMES)}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info    = mne.create_info(CH_NAMES, sfreq=fs, ch_types="eeg")
    info.set_montage(montage)
    return info


# ============================================================
# DATA LOADING
# ============================================================
def load_all_subjects(data_dir):
    """
    Expects files named: sub-XXX_ses-YY_task_motorimagery_eeg.mat
    Loads and concatenates all sessions per subject in filename order.
    Returns dict: {subject_id: {"data": [trials x ch x samples],
                                "labels": [trials]}}
    """
    raw = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".mat"):
            continue
        parts      = fname.split("_")
        subject_id = parts[0]   # sub-001

        fpath    = os.path.join(data_dir, fname)
        mat      = loadmat(fpath)
        data     = mat["data"]    # [trials x ch x samples]
        labels   = mat["labels"]  # [1 x trials] or [trials]
        if labels.ndim == 2:
            labels = labels.flatten()

        if subject_id not in raw:
            raw[subject_id] = {"data": [], "labels": []}
        raw[subject_id]["data"].append(data)
        raw[subject_id]["labels"].append(labels)

    # concatenate across sessions
    out = {}
    for sid in raw:
        X = np.concatenate(raw[sid]["data"],   axis=0)   # [trials x ch x samples]
        y = np.concatenate(raw[sid]["labels"], axis=0)   # [trials]
        out[sid] = {"data": X, "labels": y}
        print(f"[LOAD] {sid}: {X.shape[0]} trials, {X.shape[1]} ch, "
              f"L={(y==1).sum()}, R={(y==2).sum()}")
    return out


# ============================================================
# PREPROCESSING
# ============================================================
def bandpass(raw):
    """Input: [trials x ch x samples], output: same shape."""
    nyq = 0.5 * fs
    sos = sig.butter(4, [band_low/nyq, band_high/nyq],
                     btype="bandpass", output="sos")
    out = {}
    for sid, d in raw.items():
        X = d["data"].copy().astype(np.float64)
        for t in range(X.shape[0]):
            for ch in range(X.shape[1]):
                X[t, ch, :] = sig.sosfiltfilt(sos, X[t, ch, :])
        out[sid] = {"data": X, "labels": d["labels"].copy()}
    return out


def run_asr(filtered, sfreq=250, cutoff=20.0):
    """ASR calibrated on concatenated continuous signal per subject."""
    out = {}
    for sid, d in filtered.items():
        X = d["data"]   # [trials x ch x samples]
        n_trials, n_ch, n_t = X.shape

        # concatenate along time: [ch x total_samples]
        concat   = X.reshape(n_trials * n_t, n_ch).T   # wrong order: interleaves trials
        # correct: concatenate trial time axes
        concat   = X.transpose(1, 0, 2).reshape(n_ch, n_trials * n_t)

        M, T_cut = asr_calibrate(concat, sfreq=sfreq, cutoff=cutoff)
        clean    = asr_process(concat, sfreq=sfreq, M=M, T=T_cut)  # [ch x total]

        # split back into trials
        X_clean = clean.reshape(n_ch, n_trials, n_t).transpose(1, 0, 2)
        out[sid] = {"data": X_clean, "labels": d["labels"].copy()}
        print(f"[ASR] {sid} done")
    return out


def zscore_data(filtered):
    """Z-score per trial per channel."""
    out = {}
    for sid, d in filtered.items():
        X = d["data"].copy()
        for t in range(X.shape[0]):
            for ch in range(X.shape[1]):
                X[t, ch, :] = zscore(X[t, ch, :])
        out[sid] = {"data": X, "labels": d["labels"].copy()}
    return out


def build_sessions(preprocessed, n_folds=5):
    """
    Split each subject's ordered trials causally into n_folds sessions.
    Preserves L/R balance within folds since trials are already [trials x ch x T].
    Returns: {subject: {sess_id: {"X": [n x ch x T], "y": [n]}}}
    """
    sessions_all = {}
    for sid, d in preprocessed.items():
        X = d["data"]    # [trials x ch x T]
        y = d["labels"]
        n = X.shape[0]

        size     = n // n_folds
        sessions = {}
        for i in range(n_folds):
            start = i * size
            end   = n if i == n_folds - 1 else (i + 1) * size
            Xi    = X[start:end]
            yi    = y[start:end]
            if len(yi) == 0:
                continue
            sessions[i + 1] = {"X": Xi, "y": yi.astype(int)}

        sessions_all[sid] = sessions
        for s, sv in sessions.items():
            print(f"  {sid} sess {s}: X={sv['X'].shape}, "
                  f"L={(sv['y']==1).sum()}, R={(sv['y']==2).sum()}")
    return sessions_all


# ============================================================
# DISTRIBUTIONAL METRICS
# ============================================================
def rbf_kernel(X, Y, sigma):
    diff = X[:, None, :] - Y[None, :, :]
    return np.exp(-np.sum(diff**2, axis=2) / (2 * sigma**2))

def mmd_squared(X, Y, sigma):
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)
    n, m = len(X), len(Y)
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    return (Kxx.sum()/(n*(n-1)) + Kyy.sum()/(m*(m-1)) - 2*Kxy.mean())

def mmd_permutation_test(X, Y, sigma, n_perm=N_PERMUTATIONS):
    observed = mmd_squared(X, Y, sigma)
    Z        = np.vstack([X, Y])
    n        = len(X)
    null     = np.empty(n_perm)
    for i in range(n_perm):
        idx     = np.random.permutation(len(Z))
        null[i] = mmd_squared(Z[idx[:n]], Z[idx[n:]], sigma)
    return observed, (null >= observed).mean()

def median_bandwidth(X):
    from scipy.spatial.distance import pdist
    d = pdist(X)
    return float(np.median(d)) if len(d) > 0 else 1.0

def sym_kl(mu1, S1, mu2, S2, eps=1e-6):
    d    = len(mu1)
    S1   = S1 + eps*np.eye(d)
    S2   = S2 + eps*np.eye(d)
    i1   = inv(S1); i2 = inv(S2)
    diff = mu2 - mu1
    kl12 = 0.5*(np.trace(i2@S1) + diff@i2@diff - d
                + np.log(np.linalg.det(S2)/np.linalg.det(S1)))
    kl21 = 0.5*(np.trace(i1@S2) + diff@i1@diff - d
                + np.log(np.linalg.det(S1)/np.linalg.det(S2)))
    return 0.5*(kl12 + kl21)

def compute_metrics(feat_df, sess_ids, sigma):
    res = {"kl_sym":{}, "mmd_sep":{}, "mmd_sep_p":{},
           "mmd_drift":{}, "mmd_drift_p":{}}
    s1_feats = feat_df[feat_df.session==sess_ids[0]][["CSP1","CSP2"]].values

    for s in sess_ids:
        sub  = feat_df[feat_df.session==s]
        fL   = sub[sub.label==1][["CSP1","CSP2"]].values
        fR   = sub[sub.label==2][["CSP1","CSP2"]].values
        fall = sub[["CSP1","CSP2"]].values

        if len(fL) >= MIN_TRIALS_CLS and len(fR) >= MIN_TRIALS_CLS:
            res["kl_sym"][s] = sym_kl(fL.mean(0), np.cov(fL.T),
                                      fR.mean(0), np.cov(fR.T))
            v, p = mmd_permutation_test(fL, fR, sigma)
            res["mmd_sep"][s]   = v
            res["mmd_sep_p"][s] = p
        else:
            res["kl_sym"][s] = res["mmd_sep"][s] = res["mmd_sep_p"][s] = np.nan

        if s != sess_ids[0] and len(fall) >= MIN_TRIALS_CLS:
            v, p = mmd_permutation_test(s1_feats, fall, sigma)
            res["mmd_drift"][s]   = v
            res["mmd_drift_p"][s] = p

    return res


# ============================================================
# REGIME CLASSIFICATION
# ============================================================
def classify_regime(acc, mmd_drift, sess_ids):
    later = [s for s in sess_ids[1:] if s in mmd_drift]
    if len(later) < 2:
        return "ambiguous"
    acc_vals   = np.array([acc.get(s, np.nan) for s in sess_ids])
    drift_vals = np.array([mmd_drift.get(s, np.nan) for s in later])
    acc_diffs  = np.diff(acc_vals[~np.isnan(acc_vals)])
    has_jump   = np.any(acc_diffs < -0.15)
    mono_acc   = np.all(acc_diffs <= 0.02)
    dc         = drift_vals[~np.isnan(drift_vals)]
    mono_drift = len(dc) > 1 and np.all(np.diff(dc) >= 0)
    if has_jump:
        return "jump"
    if mono_acc and mono_drift:
        return "drift"
    return "ambiguous"


# ============================================================
# PER-SUBJECT ANALYSIS
# ============================================================
def analyse_subject(subject, sessions, info):
    sess_ids = sorted(sessions.keys())
    s1       = sess_ids[0]

    csp = CSP(n_components=n_csp_components, log=True,
              norm_trace=True, component_order="mutual_info")
    csp.fit(sessions[s1]["X"], sessions[s1]["y"])

    dfs = []
    for s in sess_ids:
        Xc = csp.transform(sessions[s]["X"])
        dfs.append(pd.DataFrame({
            "CSP1":    Xc[:, 0],
            "CSP2":    Xc[:, 1],
            "session": s,
            "label":   sessions[s]["y"],
        }))
    feat_df = pd.concat(dfs, ignore_index=True)

    s1_feats = feat_df[feat_df.session==s1][["CSP1","CSP2"]].values
    sigma    = median_bandwidth(s1_feats)

    metrics = compute_metrics(feat_df, sess_ids, sigma)

    clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
    m1  = feat_df.session == s1
    clf.fit(feat_df[m1][["CSP1","CSP2"]], feat_df[m1].label)
    acc = {s: (clf.predict(feat_df[feat_df.session==s][["CSP1","CSP2"]])
               == feat_df[feat_df.session==s].label).mean()
           for s in sess_ids}

    regime = classify_regime(acc, metrics["mmd_drift"], sess_ids)
    return feat_df, acc, metrics, regime, csp


# ============================================================
# FIGURES
# ============================================================
def plot_subject_summary(subject, feat_df, acc, metrics, regime, fig_dir):
    sess_ids = sorted(feat_df.session.unique())
    col      = PALETTE.get(regime, PALETTE["neutral"])

    fig = plt.figure(figsize=(16, 3.2))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.72)
    axs = [fig.add_subplot(gs[i]) for i in range(4)]

    # panel 0: CSP scatter
    ax = axs[0]
    for i, s in enumerate(sess_ids):
        for lab in [1, 2]:
            m = (feat_df.session==s) & (feat_df.label==lab)
            if not m.any(): continue
            ax.scatter(feat_df.loc[m,"CSP1"], feat_df.loc[m,"CSP2"],
                       color=_sess_color(i), marker=CLASS_MARKERS[lab],
                       s=12, alpha=0.65, linewidths=0)
    ax.set_xlabel("CSP$_1$ (log-var)")
    ax.set_ylabel("CSP$_2$ (log-var)")
    ax.set_title("Feature space")
    sess_h = [Patch(color=_sess_color(i), label=f"S{s}")
              for i, s in enumerate(sess_ids)]
    cls_h  = [Line2D([0],[0], marker=CLASS_MARKERS[l], color="k",
                     linestyle="None", markersize=5, label=CLASS_LABELS[l])
              for l in [1,2]]
    ax.legend(handles=sess_h+cls_h, ncol=2, fontsize=6, frameon=False)
    ax.grid(True)

    # panel 1: accuracy
    ax  = axs[1]
    xs  = list(range(1, len(sess_ids)+1))
    ys  = [acc.get(s, np.nan) for s in sess_ids]
    ax.plot(xs, ys, "o-", color=col, markersize=4)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Chance")
    ax.set_ylim(0, 1.05); ax.set_xticks(xs)
    ax.set_xlabel("Session"); ax.set_ylabel("Accuracy")
    ax.set_title("Classification accuracy")
    ax.legend(frameon=False); ax.grid(True)

    # panel 2: KL + MMD_sep dual axis
    import matplotlib.ticker as ticker
    ax  = axs[2]
    kl  = [metrics["kl_sym"].get(s, np.nan) for s in sess_ids]
    sep = [metrics["mmd_sep"].get(s, np.nan) for s in sess_ids]
    ax2 = ax.twinx()
    l1, = ax.plot(xs, kl,  "o--", color="#1b7837", markersize=4, label="Sym KL")
    l2, = ax2.plot(xs, sep,"s-.", color="#762a83", markersize=4, label="MMD$^2_{sep}$")
    ax.set_xlabel("Session")
    ax.set_ylabel("Sym KL", color="#1b7837", labelpad=4)
    ax2.set_ylabel("MMD$^2_{sep}$", color="#762a83", labelpad=4)
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune="both"))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune="both"))
    ax2.tick_params(axis="y", labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xticks(xs); ax.set_title("Within-session separability")
    ax.legend(handles=[l1,l2], frameon=False, fontsize=6,
              loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.grid(True)

    # panel 3: MMD drift
    ax    = axs[3]
    later = [s for s in sess_ids[1:] if s in metrics["mmd_drift"]]
    dxs   = [sess_ids.index(s)+1 for s in later]
    dvs   = [metrics["mmd_drift"][s] for s in later]
    dps   = [metrics["mmd_drift_p"][s] for s in later]
    ax.plot(dxs, dvs, "o-", color=col, markersize=4)
    for xi, yi, p in zip(dxs, dvs, dps):
        if p < 0.05:
            ax.plot(xi, yi, "*", color="#d73027", markersize=8, zorder=5)
    ax.set_xlabel("Session"); ax.set_ylabel("MMD$^2_{drift}$")
    ax.set_title("Cross-session marginal drift")
    ax.set_xticks(dxs); ax.grid(True)
    ax.annotate("* $p < 0.05$", xy=(0.97,0.04), xycoords="axes fraction",
                ha="right", fontsize=6, color="#d73027")

    fig.suptitle(f"{subject}", fontsize=9,
                 color="#333333", fontweight="bold", y=1.02)
    savefig(fig, os.path.join(fig_dir, f"{_safe(subject)}_summary.{SAVE_FMT}"))


def plot_accuracy_cohort(all_acc, regimes, fig_dir):
    """
    SHU has 25 subjects. Split by regime; within each regime sort by
    Session-1 accuracy descending. Use two column panels if n > 12.
    """
    groups = {}
    for subj, regime in regimes.items():
        groups.setdefault(regime, []).append(subj)

    regime_labels = {"jump":  "Distributional Jump",
                     "drift": "Gradual Drift",
                     "ambiguous": "Ambiguous"}

    for regime, subjs in groups.items():
        if not subjs:
            continue
        subjs = sorted(subjs,
                       key=lambda s: list(all_acc[s].values())[0]
                       if all_acc[s] else 0,
                       reverse=True)

        n     = len(subjs)
        ncols = 2 if n > 12 else 1
        half  = (n + 1) // 2
        splits= [subjs[:half], subjs[half:]] if ncols == 2 else [subjs]

        col   = PALETTE.get(regime, PALETTE["neutral"])
        fig, axes = plt.subplots(1, ncols, figsize=(6.5*ncols, 5.0), sharey=True)
        if ncols == 1:
            axes = [axes]

        cmap = plt.get_cmap("tab10")

        for ax, group in zip(axes, splits):
            max_sess = max(len(all_acc[s]) for s in group)
            xs       = list(range(1, max_sess + 1))

            for k, subj in enumerate(group):
                acc_vals = [list(all_acc[subj].values())[i]
                            if i < len(all_acc[subj]) else np.nan
                            for i in range(max_sess)]
                c = cmap(k % 10)
                ax.plot(xs, acc_vals, "o-", color=c,
                        alpha=0.75, markersize=3, linewidth=0.9)
                last_i = max((i for i, v in enumerate(acc_vals)
                              if not np.isnan(v)), default=None)
                if last_i is not None:
                    ax.text(xs[last_i] + 0.05, acc_vals[last_i],
                            subj, fontsize=5, va="center", color=c)

            ax.axhline(0.5, color="gray", ls="--", lw=0.8)
            ax.set_xticks(xs)
            ax.set_xlabel("Session")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.05)
            ax.grid(True)

        fig.suptitle(f"{regime_labels.get(regime, regime)}  (n={n})",
                     fontsize=10, fontweight="bold", color=col)
        fig.tight_layout()
        savefig(fig, os.path.join(fig_dir, f"cohort_accuracy_{regime}.{SAVE_FMT}"))


def plot_regime_bar(regimes, fig_dir):
    counts = {}
    for r in regimes.values():
        counts[r] = counts.get(r, 0) + 1
    labels = list(counts.keys())
    vals   = [counts[l] for l in labels]
    colors = [PALETTE.get(l, PALETTE["neutral"]) for l in labels]

    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(labels, vals, color=colors, width=0.45, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                str(v), ha="center", fontsize=8)
    ax.set_ylabel("Number of subjects")
    ax.set_title("Regime classification (SHU cohort)")
    ax.set_ylim(0, max(vals)*1.3)
    ax.grid(axis="y")
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, f"regime_summary_bar.{SAVE_FMT}"))


def plot_metric_trajectories(all_metrics, all_acc, regimes, fig_dir):
    """Mean +/- SEM per regime across 5 sessions."""
    regime_groups = {}
    for subj, regime in regimes.items():
        regime_groups.setdefault(regime, []).append(subj)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes = axes.flatten()
    panels = [
        ("acc",       "Accuracy",        "Classification accuracy (frozen SVM)"),
        ("kl_sym",    "Sym KL",          "Within-session class separability (KL)"),
        ("mmd_sep",   "MMD$^2_{sep}$",   "Within-session class separability (MMD)"),
        ("mmd_drift", "MMD$^2_{drift}$", "Cross-session marginal drift"),
    ]

    for ax, (key, ylabel, title) in zip(axes, panels):
        for regime, subjs in regime_groups.items():
            col = PALETTE.get(regime, PALETTE["neutral"])
            all_vals = []
            for subj in subjs:
                sess = list(all_acc[subj].keys())
                if key == "acc":
                    v = list(all_acc[subj].values())
                else:
                    v = [all_metrics[subj].get(key, {}).get(s, np.nan) for s in sess]
                all_vals.append(v)

            if not all_vals:
                continue

            max_len = max(len(v) for v in all_vals)
            mat     = np.full((len(all_vals), max_len), np.nan)
            for i, v in enumerate(all_vals):
                mat[i, :len(v)] = v

            xs   = np.arange(1, max_len+1)
            mean = np.nanmean(mat, axis=0)
            sem  = np.nanstd(mat, axis=0) / np.sqrt(np.sum(~np.isnan(mat), axis=0))

            ax.plot(xs, mean, "o-", color=col, markersize=4,
                    label=regime.capitalize())
            ax.fill_between(xs, mean-sem, mean+sem, color=col, alpha=0.15)

        if key == "acc":
            ax.axhline(0.5, color="gray", ls="--", lw=0.8)
            ax.set_ylim(0, 1.05)
        ax.set_xlabel("Session")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(True)

    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, f"cohort_metric_trajectories.{SAVE_FMT}"))


def build_results_table(all_acc, all_metrics, regimes):
    rows = []
    for subj in sorted(all_acc.keys()):
        acc_vals   = list(all_acc[subj].values())
        sess_ids   = list(all_acc[subj].keys())
        kl_vals    = [all_metrics[subj]["kl_sym"].get(s, np.nan) for s in sess_ids]
        drift_vals = [all_metrics[subj]["mmd_drift"].get(s, np.nan) for s in sess_ids[1:]]
        sig_drift  = sum(1 for s in sess_ids[1:]
                         if all_metrics[subj]["mmd_drift_p"].get(s, 1.0) < 0.05)
        rows.append({
            "Subject":        subj,
            "Regime":         regimes.get(subj, "unknown"),
            "Acc_S1":         round(acc_vals[0], 3) if acc_vals else np.nan,
            "Acc_last":       round(acc_vals[-1], 3) if acc_vals else np.nan,
            "Acc_drop":       round(acc_vals[0]-acc_vals[-1], 3) if len(acc_vals)>1 else np.nan,
            "KL_S1":          round(kl_vals[0], 3) if not np.isnan(kl_vals[0]) else np.nan,
            "KL_last":        round(kl_vals[-1], 3) if not np.isnan(kl_vals[-1]) else np.nan,
            "MMD_drift_max":  round(float(np.nanmax(drift_vals)), 4) if drift_vals else np.nan,
            "Sig_drift_sess": sig_drift,
        })
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    info = create_info()

    # Load and preprocess
    raw        = load_all_subjects(data_dir)
    filtered   = bandpass(raw)
    if USE_ASR:
        filtered = run_asr(filtered)
    normalised = zscore_data(filtered)
    del filtered

    # Build sessions via causal folding
    sessions_all = build_sessions(normalised, n_folds=N_FOLDS)
    del normalised

    all_acc     = {}
    all_metrics = {}
    regimes     = {}

    for subject in sorted(sessions_all.keys()):
        sessions = sessions_all[subject]
        if len(sessions) < 2:
            print(f"[SKIP] {subject}: fewer than 2 sessions")
            continue

        done_flag = os.path.join(fig_dir, f"{_safe(subject)}_DONE.txt")
        cache     = os.path.join(fig_dir, f"{_safe(subject)}_cache.pkl")

        if os.path.exists(done_flag) and os.path.exists(cache):
            print(f"[SKIP] {subject} (cached)")
            with open(cache, "rb") as f:
                cached = pickle.load(f)
            all_acc[subject]     = cached["acc"]
            all_metrics[subject] = cached["metrics"]
            regimes[subject]     = cached["regime"]
            continue

        print(f"\n{'='*50}\n{subject}")
        feat_df, acc, metrics, regime, csp = analyse_subject(
            subject, sessions, info)

        all_acc[subject]     = acc
        all_metrics[subject] = metrics
        regimes[subject]     = regime

        print(f"  Regime: {regime}")
        print(f"  Acc:    {acc}")

        plot_subject_summary(subject, feat_df, acc, metrics, regime, fig_dir)

        with open(cache, "wb") as f:
            pickle.dump({"acc": acc, "metrics": metrics, "regime": regime}, f)
        with open(done_flag, "w") as f:
            f.write("done\n")

    # Cohort figures
    if all_acc:
        plot_accuracy_cohort(all_acc, regimes, fig_dir)
        plot_regime_bar(regimes, fig_dir)
        plot_metric_trajectories(all_metrics, all_acc, regimes, fig_dir)

        df       = build_results_table(all_acc, all_metrics, regimes)
        csv_path = os.path.join(fig_dir, "shu_results_table.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[TABLE]\n{df.to_string(index=False)}")
        print(f"[SAVED] {csv_path}")

    print(f"\n[DONE] figures in {fig_dir}")