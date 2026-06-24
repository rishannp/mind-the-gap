# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
ALS MI Dataset - Non-stationarity Characterisation
===================================================
- Load .mat files, bandpass filter (8-30 Hz), ASR
- Split each subject's trials causally into 4 pseudo-sessions
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
from numpy.linalg import slogdet, inv

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
data_dir  = r'C:\Users\uceerjp\Desktop\PhD\Multi-Session Data\OG_Full_Data'
fig_dir   = r'C:\Users\uceerjp\Desktop\PhD\THESIS\THESIS - Chapter 3\ALS\als_nonstationarity_figures'
os.makedirs(fig_dir, exist_ok=True)

fs               = 256
band_low         = 8.0
band_high        = 30.0
asr_cutoff       = 20.0
USE_ASR          = False
n_csp_components = 2
N_FOLDS          = 4       # causal pseudo-sessions
N_PERMUTATIONS   = 1000
MIN_TRIALS_CLS   = 5       # lower than Stieger: ALS has fewer trials
SAVE_DPI         = 600
SAVE_FMT         = "png"

SUBJECT_IDS = [1, 2, 5, 9, 21, 31, 34, 39]

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
    "s1": "#2166ac", "s2": "#4dac26",
    "s3": "#d01c8b", "s4": "#f1a340",
    "drift":   "#2166ac",
    "jump":    "#d73027",
    "neutral": "#555555",
}
SESSION_COLORS = ["#2166ac", "#4dac26", "#d01c8b", "#f1a340"]
CLASS_MARKERS  = {1: "s", 2: "o"}
CLASS_LABELS   = {1: "Left", 2: "Right"}

CH_NAMES = [
    'FP1','FP2','F7','F3','FZ','F4','F8',
    'T7','C3','CZ','C4','T8',
    'P7','P3','PZ','P4','P8',
    'O1','O2'
]
CH_COORDS = np.array([
    [ 0.950,  0.309, -0.0349],
    [ 0.950, -0.309, -0.0349],
    [ 0.587,  0.809, -0.0349],
    [ 0.673,  0.545,  0.500 ],
    [ 0.719,  0.000,  0.695 ],
    [ 0.673, -0.545,  0.500 ],
    [ 0.587, -0.809, -0.0349],
    [ 6.12e-17,  0.999, -0.0349],
    [ 4.40e-17,  0.719,  0.695 ],
    [ 3.75e-33, -6.12e-17, 1.0 ],
    [ 4.40e-17, -0.719,  0.695 ],
    [ 6.12e-17, -0.999, -0.0349],
    [-0.587,  0.809, -0.0349],
    [-0.673,  0.545,  0.500 ],
    [-0.719, -8.81e-17, 0.695],
    [-0.673, -0.545,  0.500 ],
    [-0.587, -0.809, -0.0349],
    [-0.950,  0.309, -0.0349],
    [-0.950, -0.309, -0.0349],
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

def modal_len(lengths):
    vals, counts = np.unique(lengths, return_counts=True)
    return int(vals[np.argmax(counts)])

def pad_or_trim(trial, target):
    # trial: [samples x channels]
    T, ch = trial.shape
    if T > target:
        return trial[:target, :]
    if T < target:
        return np.vstack([trial, np.zeros((target - T, ch))])
    return trial


# ============================================================
# DATA LOADING
# ============================================================
def remove_trailing_zeros(lst):
    if lst and np.all(lst[-1] == 0):
        return lst[:-1]
    return lst

def load_all_subjects(data_dir, subject_ids):
    raw = {}
    for fname in os.listdir(data_dir):
        if not fname.endswith(".mat"):
            continue
        num_str = fname[len("S"):-len(".mat")]
        if not num_str.isdigit() or int(num_str) not in subject_ids:
            continue
        sid   = f"S{num_str}"
        fpath = os.path.join(data_dir, fname)
        mat   = loadmat(fpath)
        var   = f"Subject{num_str}"
        if var not in mat:
            continue
        void_arr = mat[var]
        L, R = [], []
        for item in void_arr[0]:
            L.append(item["L"])
            R.append(item["R"])
        raw[sid] = {
            "L": remove_trailing_zeros(L),
            "R": remove_trailing_zeros(R),
        }
        print(f"[LOAD] {sid}: {len(raw[sid]['L'])} L, {len(raw[sid]['R'])} R trials")
    return raw


# ============================================================
# PREPROCESSING
# ============================================================
def bandpass(trials_LR, low=8.0, high=30.0, sfreq=256):
    nyq = 0.5 * sfreq
    sos = sig.butter(4, [low/nyq, high/nyq], btype="bandpass", output="sos")
    out = {}
    for sid, data in trials_LR.items():
        out[sid] = {}
        for direction in ["L", "R"]:
            filtered = []
            for trial in data[direction]:
                # trial: [samples x channels], keep first 19 ch
                t = trial[:, :19]
                filtered.append(sig.sosfiltfilt(sos, t, axis=0))
            out[sid][direction] = filtered
    return out

def run_asr(trials_LR, sfreq=256, cutoff=20.0):
    out = {}
    for sid, data in trials_LR.items():
        all_trials  = data["L"] + data["R"]
        n_L         = len(data["L"])
        trial_lens  = [t.shape[0] for t in all_trials]

        # concatenate [samples x ch] -> transpose for ASR
        concat = np.concatenate(all_trials, axis=0).T  # [ch x total_samples]
        M, T   = asr_calibrate(concat, sfreq=sfreq, cutoff=cutoff)
        clean  = asr_process(concat, sfreq=sfreq, M=M, T=T).T  # [samples x ch]

        # split back
        cleaned, idx = [], 0
        for l in trial_lens:
            cleaned.append(clean[idx:idx+l, :])
            idx += l

        out[sid] = {
            "L": cleaned[:n_L],
            "R": cleaned[n_L:],
        }
        print(f"[ASR] {sid} done")
    return out

def causal_fold(trials, n_folds=4):
    n     = len(trials)
    size  = n // n_folds
    folds = []
    for i in range(n_folds):
        start = i * size
        end   = n if i == n_folds - 1 else (i+1) * size
        folds.append(trials[start:end])
    return folds

def build_sessions(trials_LR, n_folds=4):
    """
    Returns per-subject dict of sessions, each session:
      {"X": [n_trials x n_ch x n_samples], "y": [n_trials]}
    """
    sessions_all = {}
    for sid, data in trials_LR.items():
        L_folds = causal_fold(data["L"], n_folds)
        R_folds = causal_fold(data["R"], n_folds)

        # determine target length per subject (mode across all trials)
        all_trials = data["L"] + data["R"]
        target     = modal_len(np.array([t.shape[0] for t in all_trials]))

        sessions = {}
        for s in range(n_folds):
            trials = L_folds[s] + R_folds[s]
            labels = [1]*len(L_folds[s]) + [2]*len(R_folds[s])

            if not trials:
                continue

            # pad/trim and drop if < 50% target
            kept_t, kept_l = [], []
            for t, l in zip(trials, labels):
                if t.shape[0] < 0.5 * target:
                    continue
                kept_t.append(pad_or_trim(t, target))
                kept_l.append(l)

            if not kept_t:
                continue

            # [n_trials x n_ch x n_samples] (transpose from samples x ch)
            X = np.stack([t.T for t in kept_t])
            y = np.array(kept_l, dtype=int)
            sessions[s+1] = {"X": X, "y": y}

        sessions_all[sid] = sessions
        for s, sv in sessions.items():
            print(f"  {sid} sess {s}: X={sv['X'].shape}, "
                  f"L={(sv['y']==1).sum()}, R={(sv['y']==2).sum()}")

    return sessions_all


# ============================================================
# DISTRIBUTIONAL METRICS  (identical to Stieger script)
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
    Z = np.vstack([X, Y])
    n = len(X)
    null = np.empty(n_perm)
    for i in range(n_perm):
        idx     = np.random.permutation(len(Z))
        null[i] = mmd_squared(Z[idx[:n]], Z[idx[n:]], sigma)
    return observed, (null >= observed).mean()

def median_bandwidth(X):
    from scipy.spatial.distance import pdist
    d = pdist(X)
    return float(np.median(d)) if len(d) > 0 else 1.0

def sym_kl(mu1, S1, mu2, S2, eps=1e-6):
    d   = len(mu1)
    S1  = S1 + eps*np.eye(d)
    S2  = S2 + eps*np.eye(d)
    i1  = inv(S1); i2 = inv(S2)
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
    ALS has 8 subjects. No need to split into columns.
    One figure per regime, subjects colour-coded, labelled inline.
    """
    groups = {}
    for subj, regime in regimes.items():
        groups.setdefault(regime, []).append(subj)

    regime_labels = {"jump":"Distributional Jump","drift":"Gradual Drift",
                     "ambiguous":"Ambiguous"}

    for regime, subjs in groups.items():
        if not subjs:
            continue
        subjs = sorted(subjs, key=lambda s: list(all_acc[s].values())[0],
                       reverse=True)
        col   = PALETTE.get(regime, PALETTE["neutral"])

        fig, ax = plt.subplots(figsize=(6, 4))
        cmap    = plt.get_cmap("tab10")

        for k, subj in enumerate(subjs):
            acc_vals = list(all_acc[subj].values())
            xs       = list(range(1, len(acc_vals)+1))
            c        = cmap(k % 10)
            ax.plot(xs, acc_vals, "o-", color=c, markersize=4,
                    linewidth=1.0, alpha=0.85)
            ax.text(xs[-1]+0.05, acc_vals[-1], subj,
                    fontsize=7, va="center", color=c)

        ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.set_xticks([1,2,3,4])
        ax.set_xlabel("Session")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{regime_labels.get(regime, regime)}  (n={len(subjs)})",
                     color=col, fontweight="bold")
        ax.grid(True)
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
    ax.set_title("Regime classification (ALS cohort)")
    ax.set_ylim(0, max(vals)*1.3)
    ax.grid(axis="y")
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, f"regime_summary_bar.{SAVE_FMT}"))


def plot_metric_trajectories(all_metrics, all_acc, regimes, fig_dir):
    """Mean +/- SEM per regime across 4 sessions."""
    regime_groups = {}
    for subj, regime in regimes.items():
        regime_groups.setdefault(regime, []).append(subj)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes = axes.flatten()
    panels = [
        ("acc",       "Accuracy",         "Classification accuracy (frozen SVM)"),
        ("kl_sym",    "Sym KL",           "Within-session class separability (KL)"),
        ("mmd_sep",   "MMD$^2_{sep}$",    "Within-session class separability (MMD)"),
        ("mmd_drift", "MMD$^2_{drift}$",  "Cross-session marginal drift"),
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

    # Load
    raw = load_all_subjects(data_dir, SUBJECT_IDS)

    # Bandpass
    filtered = bandpass(raw)

    # ASR
    if USE_ASR:
        filtered = run_asr(filtered)

    # Build sessions via causal folding
    sessions_all = build_sessions(filtered)

    all_acc     = {}
    all_metrics = {}
    regimes     = {}

    for subject in sorted(sessions_all.keys()):
        sessions = sessions_all[subject]
        if len(sessions) < 2:
            print(f"[SKIP] {subject}: fewer than 2 sessions")
            continue

        print(f"\n{'='*50}\n{subject}")

        # z-score per trial
        for s in sessions:
            sessions[s]["X"] = zscore(sessions[s]["X"], axis=2)

        feat_df, acc, metrics, regime, csp = analyse_subject(subject, sessions, info)

        all_acc[subject]     = acc
        all_metrics[subject] = metrics
        regimes[subject]     = regime

        print(f"  Regime: {regime}")
        print(f"  Acc: {acc}")

        plot_subject_summary(subject, feat_df, acc, metrics, regime, fig_dir)

    # Cohort figures
    if all_acc:
        plot_accuracy_cohort(all_acc, regimes, fig_dir)
        plot_regime_bar(regimes, fig_dir)
        plot_metric_trajectories(all_metrics, all_acc, regimes, fig_dir)

        df = build_results_table(all_acc, all_metrics, regimes)
        csv_path = os.path.join(fig_dir, "als_results_table.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[TABLE]\n{df.to_string(index=False)}")
        print(f"[SAVED] {csv_path}")

    print(f"\n[DONE] figures in {fig_dir}")