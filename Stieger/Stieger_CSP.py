# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import os
os.environ["MNE_BROWSER_BACKEND"] = "matplotlib"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib
matplotlib.use("Agg")

import re
import pickle
import numpy as np
import pandas as pd
import scipy.signal as sig
from scipy.stats import zscore
from scipy.linalg import sqrtm
import mne
from mne.decoding import CSP
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ============================================================
# CONFIG
# ============================================================
server_dir   = "/scratch/uceerjp"
fig_dir      = os.path.join(server_dir, "csp_nonstationarity_figures")
os.makedirs(fig_dir, exist_ok=True)

fs              = 1000
band_low        = 8.0
band_high       = 30.0
asr_cutoff      = 20.0
USE_ASR         = False
n_csp_components = 2
USE_MOTOR_SUBSET = True
N_PERMUTATIONS  = 1000   # MMD permutation test
MIN_TRIALS_CLS  = 15     # minimum trials per class for KL / MMD
SAVE_DPI        = 600
SAVE_FMT        = "png"

# -- Publication style --------------------------------------
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif"],
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  7,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.linewidth":   0.8,
    "xtick.major.width":0.8,
    "ytick.major.width":0.8,
    "grid.linewidth":   0.5,
    "grid.alpha":       0.35,
    "lines.linewidth":  1.2,
})

PALETTE = {
    "s1": "#2166ac",
    "s2": "#4dac26",
    "s3": "#d01c8b",
    "s4": "#f1a340",
    "drift": "#2166ac",
    "jump":  "#d73027",
    "neutral":"#555555",
}

# ============================================================
# ELECTRODES
# ============================================================
FULL_ELECTRODE_LABELS = [
    'FP1','FPZ','FP2','AF3','AF4',
    'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
    'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
    'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
    'P7','P5','P3','P1','PZ','P2','P4','P6','P8',
    'PO7','PO5','PO3','POZ','PO4','PO6','PO8',
    'CB1','O1','OZ','O2','CB2'
]

FULL_XYZ_COORDS = np.array([
    [-0.0334442687034607, 0.1260661125183110, 0.0350484585762024],
    [ 0.0055746710300446, 0.1327504539489750, 0.0318808817863464],
    [ 0.0405898237228394, 0.1194330024719240, 0.0381312108039856],
    [-0.0442471599578857, 0.1160607051849370, 0.0607592487335205],
    [ 0.0471005964279175, 0.1088930892944340, 0.0639779710769653],
    [-0.0844468021392822, 0.0796512651443481, 0.0335807991027832],
    [-0.0734652090072632, 0.0887112236022949, 0.0566550970077515],
    [-0.0542749929428101, 0.0978937625885010, 0.0797256994247437],
    [-0.0298427033424377, 0.0968624114990234, 0.0974580574035645],
    [ 0.0004851793870330, 0.0947946071624756, 0.1012303543090820],
    [ 0.0352490544319153, 0.0907062435150147, 0.0946333217620850],
    [ 0.0600456047058105, 0.0869272518157959, 0.0823328304290771],
    [ 0.0797356843948364, 0.0813320636749268, 0.0654329872131348],
    [ 0.0840118026733398, 0.0684696769714356, 0.0431844949722290],
    [-0.0914600467681885, 0.0486501407623291, 0.0361480712890625],
    [-0.0873526382446289, 0.0616052007675171, 0.0687911653518677],
    [-0.0687758874893188, 0.0669712018966675, 0.0915870666503906],
    [-0.0367527508735657, 0.0751429653167725, 0.1129959774017330],
    [ 0.0014420005679131, 0.0721511554718018, 0.1245254516601560],
    [ 0.0362721490859985, 0.0629681825637817, 0.1171550273895260],
    [ 0.0672762870788574, 0.0590549325942993, 0.1002188968658450],
    [ 0.0893869686126709, 0.0512503623962402, 0.0756169891357422],
    [ 0.0939384746551514, 0.0361423254013062, 0.0441341543197632],
    [-0.0980308723449707, 0.0236502599716187, 0.0393873858451843],
    [-0.0920487689971924, 0.0287513732910156, 0.0740070772171021],
    [-0.0751492357254028, 0.0362352824211121, 0.1056236267089840],
    [-0.0428757858276367, 0.0423967075347900, 0.1274349975585940],
    [-0.0023281142115593, 0.0403706359863281, 0.1425280857086180],
    [ 0.0380204963684082, 0.0345760297775269, 0.1357659435272220],
    [ 0.0725486469268799, 0.0255098009109497, 0.1174332237243650],
    [ 0.0961302852630615, 0.0195411980152130, 0.0876507568359375],
    [ 0.0987967014312744, 0.00888039290905099, 0.0508006238937378],
    [-0.0944289016723633, -0.00871051907539368, 0.0500650882720947],
    [-0.0897719097137451, -0.00418301045894623, 0.0814741992950439],
    [-0.0775259637832642, -0.00126781433820725, 0.1174706363677980],
    [-0.0445305395126343, 0.00404738843441010, 0.1375082969665530],
    [-0.0062751847505570, -0.00109832718968391, 0.1514622783660890],
    [ 0.0371585178375244, -0.00337867230176926, 0.1507100296020510],
    [ 0.0741236543655396, -0.0144876587390900, 0.1286501312255860],
    [ 0.0930694580078125, -0.0169668614864349, 0.0952802944183350],
    [ 0.0973999118804932, -0.0216706252098084, 0.0622142219543457],
    [-0.0861686801910400, -0.0348205828666687, 0.0589149141311646],
    [-0.0803685951232910, -0.0368253207206726, 0.0850807189941406],
    [-0.0666205739974976, -0.0366869902610779, 0.1123663711547850],
    [-0.0398402833938599, -0.0365661621093750, 0.1353108119964600],
    [-0.0099115717411041, -0.0364873552322388, 0.1464972782135010],
    [ 0.0323527789115906, -0.0419814348220825, 0.1399067211151120],
    [ 0.0548041343688965, -0.0453798246383667, 0.1214561939239500],
    [ 0.0882028770446777, -0.0529665231704712, 0.0995615196228027],
    [ 0.0832171440124512, -0.0458828687667847, 0.0602024412155151],
    [-0.0753398323059082, -0.0593344116210938, 0.0652968883514404],
    [-0.0658427429199219, -0.0612779426574707, 0.0900853157043457],
    [-0.0488899660110474, -0.0634949159622192, 0.1095213794708250],
    [-0.0098540723323822, -0.0645778799057007, 0.1204350185394290],
    [ 0.0316038441658020, -0.0687834930419922, 0.1148945426940920],
    [ 0.0524488925933838, -0.0722089910507202, 0.0978014659881592],
    [ 0.0669306993484497, -0.0723361635208130, 0.0745094013214111],
    [-0.0433073425292969, -0.0864870071411133, 0.0612648391723633],
    [-0.0466513824462891, -0.0814486694335938, 0.0849731540679932],
    [-0.0092247486114502, -0.0849366664886475, 0.0929535102844238],
    [ 0.0289490580558777, -0.0832236099243164, 0.0917889118194580],
    [ 0.0252702856063843, -0.0942262458801270, 0.0652664709091187],
])


def select_motor_channels(labels, coords):
    keep_idx, keep_labels = [], []
    for i, ch in enumerate(labels):
        ch_u = ch.upper()
        if ch_u.startswith(("FP","AF","FT","TP","PO","CB","O")):
            continue
        if ch_u.startswith(("F","FC","C","CP","P")):
            keep_idx.append(i)
            keep_labels.append(ch)
    if not keep_idx:
        raise RuntimeError("No motor channels matched.")
    return keep_labels, coords[np.array(keep_idx)], np.array(keep_idx)


if USE_MOTOR_SUBSET:
    electrode_labels, xyz_coords, keep_idx = select_motor_channels(
        FULL_ELECTRODE_LABELS, FULL_XYZ_COORDS)
else:
    electrode_labels = list(FULL_ELECTRODE_LABELS)
    xyz_coords       = FULL_XYZ_COORDS.copy()
    keep_idx         = np.arange(len(FULL_ELECTRODE_LABELS))

print(f"[CH] n_channels={len(electrode_labels)}, motor_subset={USE_MOTOR_SUBSET}")


# ============================================================
# HELPERS
# ============================================================
def _safe(s):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(s))

def savefig(fig, path):
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

def bandpass(X):
    nyq = 0.5 * fs
    sos = sig.butter(4, [band_low/nyq, band_high/nyq], btype="bandpass", output="sos")
    return sig.sosfiltfilt(sos, X, axis=2)

def zscore_trials(X):
    return zscore(X, axis=2)

def modal_len(lengths):
    vals, counts = np.unique(lengths, return_counts=True)
    return int(vals[np.argmax(counts)])

def pad_or_trim(eeg, target):
    ch, T = eeg.shape
    if T > target:
        return eeg[:, :target]
    if T < target:
        return np.concatenate([eeg, np.zeros((ch, target - T))], axis=1)
    return eeg

def create_info():
    ch_pos  = {c: xyz_coords[i] for i, c in enumerate(electrode_labels)}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info    = mne.create_info(electrode_labels, sfreq=fs, ch_types="eeg")
    info.set_montage(montage)
    return info

def parse_fname(fname):
    m = re.match(r"^(S\d+)_Session_(\d+)\.pkl$", fname)
    return (m.group(1), int(m.group(2))) if m else (None, None)


# ============================================================
# DATA LOADING
# ============================================================
def index_files():
    files = [f for f in os.listdir(server_dir)
             if f.startswith("S") and f.endswith(".pkl")]
    subj_map = {}
    for f in files:
        subj, sess = parse_fname(f)
        if subj is None:
            continue
        subj_map.setdefault(subj, []).append((sess, os.path.join(server_dir, f)))
    for s in subj_map:
        subj_map[s] = sorted(subj_map[s])
    return subj_map


def load_subject(file_list):
    sessions = {}
    for sess_id, path in file_list:
        with open(path, "rb") as f:
            bci = pickle.load(f)

        trials, labels = [], []
        for trial, meta in zip(bci["data"], bci["TrialData"]):
            lab = meta.get("targetnumber")
            if lab not in [1, 2]:
                continue
            eeg = np.asarray(trial, dtype=np.float64)
            if eeg.shape[0] != len(FULL_ELECTRODE_LABELS):
                eeg = eeg.T
            if eeg.shape[0] != len(FULL_ELECTRODE_LABELS):
                continue
            trials.append(eeg[keep_idx, :])
            labels.append(lab)

        if not trials:
            continue

        target = modal_len(np.array([t.shape[1] for t in trials]))
        X = np.stack([pad_or_trim(t, target) for t in trials])
        y = np.asarray(labels, dtype=int)
        sessions[sess_id] = {"X": X, "y": y}
        print(f"  sess {sess_id}: X={X.shape}, L={(y==1).sum()}, R={(y==2).sum()}")

    return dict(sorted(sessions.items()))


def enforce_target_len(sessions):
    lens   = [s["X"].shape[2] for s in sessions.values()]
    target = modal_len(np.array(lens))
    for sid in sessions:
        X = sessions[sid]["X"]
        T = X.shape[2]
        if T > target:
            sessions[sid]["X"] = X[:, :, :target]
        elif T < target:
            pad = np.zeros((X.shape[0], X.shape[1], target - T))
            sessions[sid]["X"] = np.concatenate([X, pad], axis=2)


# ============================================================
# DISTRIBUTIONAL METRICS
# ============================================================
def rbf_kernel_matrix(X, Y, sigma):
    # X: [n, d], Y: [m, d]
    diff = X[:, None, :] - Y[None, :, :]           # [n, m, d]
    return np.exp(-np.sum(diff**2, axis=2) / (2 * sigma**2))

def mmd_squared(X, Y, sigma):
    Kxx = rbf_kernel_matrix(X, X, sigma)
    Kyy = rbf_kernel_matrix(Y, Y, sigma)
    Kxy = rbf_kernel_matrix(X, Y, sigma)
    n, m = len(X), len(Y)
    # unbiased estimator
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    return (Kxx.sum() / (n*(n-1)) + Kyy.sum() / (m*(m-1))
            - 2 * Kxy.mean())

def mmd_permutation_test(X, Y, sigma, n_perm=N_PERMUTATIONS):
    observed = mmd_squared(X, Y, sigma)
    Z = np.vstack([X, Y])
    n = len(X)
    null = np.empty(n_perm)
    for i in range(n_perm):
        idx    = np.random.permutation(len(Z))
        null[i] = mmd_squared(Z[idx[:n]], Z[idx[n:]], sigma)
    p = (null >= observed).mean()
    return observed, p

def median_bandwidth(X):
    # median heuristic on pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(X)
    return np.median(d) if len(d) > 0 else 1.0

def sym_kl_gaussian(mu1, S1, mu2, S2, eps=1e-6):
    d  = len(mu1)
    S1 = S1 + eps * np.eye(d)
    S2 = S2 + eps * np.eye(d)
    S2inv = np.linalg.inv(S2)
    S1inv = np.linalg.inv(S1)
    diff  = mu2 - mu1
    kl12  = 0.5 * (np.trace(S2inv @ S1)
                   + diff @ S2inv @ diff - d
                   + np.log(np.linalg.det(S2) / np.linalg.det(S1)))
    kl21  = 0.5 * (np.trace(S1inv @ S2)
                   + diff @ S1inv @ diff - d
                   + np.log(np.linalg.det(S1) / np.linalg.det(S2)))
    return 0.5 * (kl12 + kl21)

def compute_metrics(feat_df, sess_ids, sigma):
    """
    Returns dict with keys:
      kl_sym[s]        : within-session sym KL between L and R
      mmd_sep[s]       : within-session MMD between L and R
      mmd_sep_p[s]     : p-value
      mmd_drift[s]     : cross-session MMD vs session 1 (s>1)
      mmd_drift_p[s]   : p-value
    """
    results = {
        "kl_sym":    {},
        "mmd_sep":   {},
        "mmd_sep_p": {},
        "mmd_drift":   {},
        "mmd_drift_p": {},
    }

    s1_feats = feat_df[feat_df.session == sess_ids[0]][["CSP1","CSP2"]].values

    for s in sess_ids:
        sub  = feat_df[feat_df.session == s]
        fL   = sub[sub.label == 1][["CSP1","CSP2"]].values
        fR   = sub[sub.label == 2][["CSP1","CSP2"]].values
        fall = sub[["CSP1","CSP2"]].values

        # ---- sym KL (requires >= MIN_TRIALS_CLS per class)
        if len(fL) >= MIN_TRIALS_CLS and len(fR) >= MIN_TRIALS_CLS:
            results["kl_sym"][s] = sym_kl_gaussian(
                fL.mean(0), np.cov(fL.T),
                fR.mean(0), np.cov(fR.T))
        else:
            results["kl_sym"][s] = np.nan

        # ---- within-session MMD (separability)
        if len(fL) >= MIN_TRIALS_CLS and len(fR) >= MIN_TRIALS_CLS:
            v, p = mmd_permutation_test(fL, fR, sigma)
            results["mmd_sep"][s]   = v
            results["mmd_sep_p"][s] = p
        else:
            results["mmd_sep"][s]   = np.nan
            results["mmd_sep_p"][s] = np.nan

        # ---- cross-session drift (marginal, vs session 1)
        if s != sess_ids[0] and len(fall) >= MIN_TRIALS_CLS:
            v, p = mmd_permutation_test(s1_feats, fall, sigma)
            results["mmd_drift"][s]   = v
            results["mmd_drift_p"][s] = p

    return results


# ============================================================
# REGIME CLASSIFICATION
# ============================================================
def classify_regime(acc, mmd_drift, sess_ids):
    """
    Gradual drift  : accuracy monotonically decreases AND drift smoothly increases.
    Distributional jump : step-change drop in accuracy between adjacent sessions
                          accompanied by a large discrete MMD increase.
    Returns 'jump', 'drift', or 'ambiguous'.
    """
    later = [s for s in sess_ids[1:] if s in mmd_drift]
    if len(later) < 2:
        return "ambiguous"

    acc_vals   = np.array([acc.get(s, np.nan) for s in sess_ids])
    drift_vals = np.array([mmd_drift.get(s, np.nan) for s in later])

    # Consecutive accuracy drops
    acc_diffs = np.diff(acc_vals[~np.isnan(acc_vals)])

    # Step-change: any single-step accuracy drop > 15 pp
    has_jump = np.any(acc_diffs < -0.15)

    # Monotone degradation: all diffs <= 0 and drift strictly increasing
    monotone_acc   = np.all(acc_diffs <= 0.02)   # allow 2pp noise
    drift_clean    = drift_vals[~np.isnan(drift_vals)]
    monotone_drift = len(drift_clean) > 1 and np.all(np.diff(drift_clean) >= 0)

    if has_jump:
        return "jump"
    if monotone_acc and monotone_drift:
        return "drift"
    return "ambiguous"


# ============================================================
# PER-SUBJECT ANALYSIS
# ============================================================
def analyse_subject(subject, sessions, info):
    sess_ids = sorted(sessions.keys())
    s1 = sess_ids[0]

    Xtr, ytr = sessions[s1]["X"], sessions[s1]["y"]

    # Fit CSP on session 1
    csp = CSP(n_components=n_csp_components, log=True,
              norm_trace=True, component_order="mutual_info")
    csp.fit(Xtr, ytr)

    # Project all sessions
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

    # Fix bandwidth on session 1 features
    s1_feats = feat_df[feat_df.session == s1][["CSP1","CSP2"]].values
    sigma    = median_bandwidth(s1_feats)

    # Metrics
    metrics = compute_metrics(feat_df, sess_ids, sigma)

    # SVM accuracy (frozen, trained on session 1)
    clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
    m1  = feat_df.session == s1
    clf.fit(feat_df[m1][["CSP1","CSP2"]], feat_df[m1].label)
    acc = {s: (clf.predict(feat_df[feat_df.session==s][["CSP1","CSP2"]])
               == feat_df[feat_df.session==s].label).mean()
           for s in sess_ids}

    regime = classify_regime(acc, metrics["mmd_drift"], sess_ids)

    return feat_df, acc, metrics, regime, csp, sigma


# ============================================================
# FIGURES
# ============================================================
SESSION_COLORS  = ["#2166ac","#4dac26","#d01c8b","#f1a340"]
CLASS_MARKERS   = {1: "s", 2: "o"}
CLASS_LABELS    = {1: "Left", 2: "Right"}

def _sess_color(i):
    return SESSION_COLORS[i % len(SESSION_COLORS)]

def plot_subject_summary(subject, feat_df, acc, metrics, regime, fig_dir):
    """
    4-panel per-subject summary:
      [0] CSP scatter coloured by session
      [1] Accuracy across sessions
      [2] Sym KL divergence + MMD_sep across sessions
      [3] MMD_drift (cross-session) across sessions
    """
    sess_ids = sorted(feat_df.session.unique())
    col      = PALETTE["jump"] if regime == "jump" else (
               PALETTE["drift"] if regime == "drift" else PALETTE["neutral"])

    fig = plt.figure(figsize=(12, 3.2))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.42)
    axs = [fig.add_subplot(gs[i]) for i in range(4)]

    # -- panel 0: scatter -------------------------------------
    ax = axs[0]
    for i, s in enumerate(sess_ids):
        for lab in [1, 2]:
            m = (feat_df.session == s) & (feat_df.label == lab)
            if not m.any():
                continue
            ax.scatter(feat_df.loc[m,"CSP1"], feat_df.loc[m,"CSP2"],
                       color=_sess_color(i), marker=CLASS_MARKERS[lab],
                       s=12, alpha=0.65, linewidths=0)
    ax.set_xlabel("CSP$_1$ (log-var)")
    ax.set_ylabel("CSP$_2$ (log-var)")
    ax.set_title("Feature space")
    sess_handles = [Patch(color=_sess_color(i), label=f"S{s}")
                    for i, s in enumerate(sess_ids)]
    cls_handles  = [Line2D([0],[0], marker=CLASS_MARKERS[l], color="k",
                           linestyle="None", markersize=5,
                           label=CLASS_LABELS[l]) for l in [1,2]]
    ax.legend(handles=sess_handles + cls_handles, ncol=2,
              fontsize=6, frameon=False)
    ax.grid(True)

    # -- panel 1: accuracy ------------------------------------
    ax = axs[1]
    xs  = list(range(1, len(sess_ids)+1))
    ys  = [acc.get(s, np.nan) for s in sess_ids]
    ax.plot(xs, ys, "o-", color=col, markersize=4)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Chance")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(xs)
    ax.set_xlabel("Session")
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification accuracy")
    ax.legend(frameon=False)
    ax.grid(True)

    # -- panel 2: KL + MMD_sep --------------------------------
    ax = axs[2]
    kl_vals  = [metrics["kl_sym"].get(s, np.nan) for s in sess_ids]
    sep_vals = [metrics["mmd_sep"].get(s, np.nan) for s in sess_ids]
    ax2      = ax.twinx()
    l1, = ax.plot(xs, kl_vals,  "o--", color="#1b7837", markersize=4, label="Sym KL")
    l2, = ax2.plot(xs, sep_vals,"s-.", color="#762a83", markersize=4, label="MMD$_{sep}$")
    ax.set_xlabel("Session")
    ax.set_ylabel("Sym KL", color="#1b7837")
    ax2.set_ylabel("MMD$^2_{sep}$", color="#762a83")
    ax.set_xticks(xs)
    ax.set_title("Within-session separability")
    ax.legend(handles=[l1,l2], frameon=False, fontsize=6)
    ax.grid(True)

    # -- panel 3: MMD_drift -----------------------------------
    ax = axs[3]
    later       = [s for s in sess_ids[1:] if s in metrics["mmd_drift"]]
    drift_xs    = [sess_ids.index(s)+1 for s in later]
    drift_vals  = [metrics["mmd_drift"][s] for s in later]
    drift_p     = [metrics["mmd_drift_p"][s] for s in later]
    ax.plot(drift_xs, drift_vals, "o-", color=col, markersize=4)
    # mark significant drift
    for xi, yi, p in zip(drift_xs, drift_vals, drift_p):
        if p < 0.05:
            ax.plot(xi, yi, "*", color="#d73027", markersize=8, zorder=5)
    ax.set_xlabel("Session")
    ax.set_ylabel("MMD$^2_{drift}$")
    ax.set_title("Cross-session marginal drift")
    ax.set_xticks(drift_xs)
    ax.grid(True)
    ax.annotate("* $p < 0.05$", xy=(0.97,0.04), xycoords="axes fraction",
                ha="right", fontsize=6, color="#d73027")

    fig.suptitle(f"{subject}  |  regime: {regime}", fontsize=9,
                 color=col, fontweight="bold", y=1.02)
    savefig(fig, os.path.join(fig_dir, f"{_safe(subject)}_summary.{SAVE_FMT}"))


def plot_accuracy_cohort(all_acc, regimes, sess_ids_map, fig_dir):
    """
    Split 62 subjects by regime (jump / drift / ambiguous).
    Within each regime group, sort by Session-1 accuracy descending.
    One figure per regime group, max 31 subjects per panel column.
    """
    groups = {}
    for subj, regime in regimes.items():
        groups.setdefault(regime, []).append(subj)

    regime_order = ["jump", "drift", "ambiguous"]
    regime_labels = {"jump": "Distributional Jump", "drift": "Gradual Drift",
                     "ambiguous": "Ambiguous"}

    for regime in regime_order:
        subjs = groups.get(regime, [])
        if not subjs:
            continue

        # Sort by session-1 accuracy descending
        subjs = sorted(subjs,
                       key=lambda s: list(all_acc[s].values())[0]
                       if all_acc[s] else 0,
                       reverse=True)

        n     = len(subjs)
        ncols = 2 if n > 16 else 1
        half  = (n + 1) // 2 if ncols == 2 else n
        splits= [subjs[:half], subjs[half:]] if ncols == 2 else [subjs]

        fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5.5),
                                 sharey=True)
        if ncols == 1:
            axes = [axes]

        col = PALETTE.get(regime, PALETTE["neutral"])

        for ax, group in zip(axes, splits):
            max_sess = max(len(all_acc[s]) for s in group)
            xs       = list(range(1, max_sess + 1))

            for subj in group:
                acc_vals = [list(all_acc[subj].values())[i]
                            if i < len(all_acc[subj]) else np.nan
                            for i in range(max_sess)]
                ax.plot(xs, acc_vals, "o-", color=col,
                        alpha=0.45, markersize=3, linewidth=0.9)
                # label at last valid point
                last_i = max((i for i, v in enumerate(acc_vals)
                              if not np.isnan(v)), default=None)
                if last_i is not None:
                    ax.text(xs[last_i] + 0.05, acc_vals[last_i],
                            subj, fontsize=5, va="center", color="#333333")

            ax.axhline(0.5, color="gray", ls="--", lw=0.8)
            ax.set_xticks(xs)
            ax.set_xlabel("Session")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.05)
            ax.grid(True)

        fig.suptitle(f"{regime_labels[regime]}  (n={n})",
                     fontsize=10, fontweight="bold", color=col)
        fig.tight_layout()
        savefig(fig, os.path.join(fig_dir, f"cohort_accuracy_{regime}.{SAVE_FMT}"))


def plot_regime_summary_bar(regimes, fig_dir):
    counts = {"jump": 0, "drift": 0, "ambiguous": 0}
    for r in regimes.values():
        counts[r] = counts.get(r, 0) + 1

    labels = list(counts.keys())
    vals   = [counts[l] for l in labels]
    colors = [PALETTE.get(l, PALETTE["neutral"]) for l in labels]

    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(v), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Number of subjects")
    ax.set_title("Non-stationarity regime classification")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.grid(axis="y")
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, f"regime_summary_bar.{SAVE_FMT}"))


def plot_metric_trajectories_cohort(all_metrics, all_acc, regimes, fig_dir):
    """
    Mean +/- SEM trajectories per regime for:
      - accuracy, sym KL, MMD_sep, MMD_drift
    All on one 2x2 figure.
    """
    regime_groups = {}
    for subj, regime in regimes.items():
        regime_groups.setdefault(regime, []).append(subj)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes      = axes.flatten()
    metrics_to_plot = [
        ("acc",       "Accuracy",               "Classification accuracy (frozen SVM)"),
        ("kl_sym",    "Sym KL divergence",       "Within-session class separability (KL)"),
        ("mmd_sep",   "MMD$^2_{sep}$",           "Within-session class separability (MMD)"),
        ("mmd_drift", "MMD$^2_{drift}$",         "Cross-session marginal drift (vs Session 1)"),
    ]

    for ax, (key, ylabel, title) in zip(axes, metrics_to_plot):
        for regime, subjs in regime_groups.items():
            col = PALETTE.get(regime, PALETTE["neutral"])

            # gather per-subject session vectors
            all_vals = []
            for subj in subjs:
                if key == "acc":
                    src = all_acc[subj]
                    v   = list(src.values())
                else:
                    src = all_metrics[subj].get(key, {})
                    # use ordered session keys from acc
                    sess = list(all_acc[subj].keys())
                    v    = [src.get(s, np.nan) for s in sess]
                all_vals.append(v)

            if not all_vals:
                continue

            max_len = max(len(v) for v in all_vals)
            mat     = np.full((len(all_vals), max_len), np.nan)
            for i, v in enumerate(all_vals):
                mat[i, :len(v)] = v

            xs   = np.arange(1, max_len + 1)
            mean = np.nanmean(mat, axis=0)
            sem  = np.nanstd(mat, axis=0) / np.sqrt(np.sum(~np.isnan(mat), axis=0))

            ax.plot(xs, mean, "o-", color=col, markersize=4,
                    label=regime.capitalize())
            ax.fill_between(xs, mean - sem, mean + sem,
                            color=col, alpha=0.15)

        if key == "acc":
            ax.axhline(0.5, color="gray", ls="--", lw=0.8)
            ax.set_ylim(0, 1.05)
        if key == "mmd_drift":
            ax.set_xticks(np.arange(2, max_len + 1))
            ax.set_xlabel("Session (vs Session 1)")
        else:
            ax.set_xlabel("Session")

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(True)

    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, f"cohort_metric_trajectories.{SAVE_FMT}"))


def build_results_table(all_acc, all_metrics, regimes):
    rows = []
    for subj in sorted(all_acc.keys(), key=lambda s: int(s[1:])):
        acc_vals   = list(all_acc[subj].values())
        sess_ids   = list(all_acc[subj].keys())
        kl_vals    = [all_metrics[subj]["kl_sym"].get(s, np.nan) for s in sess_ids]
        drift_vals = [all_metrics[subj]["mmd_drift"].get(s, np.nan) for s in sess_ids[1:]]

        # Significant drift sessions
        sig_drift = sum(
            1 for s in sess_ids[1:]
            if all_metrics[subj]["mmd_drift_p"].get(s, 1.0) < 0.05
        )

        rows.append({
            "Subject":         subj,
            "Regime":          regimes.get(subj, "unknown"),
            "Acc_S1":          round(acc_vals[0], 3) if acc_vals else np.nan,
            "Acc_last":        round(acc_vals[-1], 3) if acc_vals else np.nan,
            "Acc_drop":        round(acc_vals[0] - acc_vals[-1], 3) if len(acc_vals)>1 else np.nan,
            "KL_S1":           round(kl_vals[0], 3) if not np.isnan(kl_vals[0]) else np.nan,
            "KL_last":         round(kl_vals[-1], 3) if not np.isnan(kl_vals[-1]) else np.nan,
            "MMD_drift_max":   round(float(np.nanmax(drift_vals)), 4) if drift_vals else np.nan,
            "Sig_drift_sess":  sig_drift,
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    info       = create_info()
    subj_files = index_files()

    all_acc     = {}
    all_metrics = {}
    regimes     = {}

    for subject in sorted(subj_files, key=lambda s: int(s[1:])):
        done_flag = os.path.join(fig_dir, f"{_safe(subject)}_DONE.txt")
        if os.path.exists(done_flag):
            print(f"[SKIP] {subject}")
            # reload cached results if needed
            cache = os.path.join(fig_dir, f"{_safe(subject)}_cache.pkl")
            if os.path.exists(cache):
                with open(cache,"rb") as f:
                    cached = pickle.load(f)
                all_acc[subject]     = cached["acc"]
                all_metrics[subject] = cached["metrics"]
                regimes[subject]     = cached["regime"]
            continue

        print(f"\n{'='*50}\n{subject}")
        sessions = load_subject(subj_files[subject])
        if len(sessions) < 2:
            continue

        enforce_target_len(sessions)
        for s in sessions:
            sessions[s]["X"] = bandpass(sessions[s]["X"])
        if USE_ASR:
            from asrpy.asr import asr_calibrate, asr_process
            concat = np.concatenate([
                np.concatenate([sessions[s]["X"][i] for i in range(sessions[s]["X"].shape[0])], axis=1)
                for s in sessions], axis=1)
            M, Tcut = asr_calibrate(concat, sfreq=fs, cutoff=asr_cutoff)
            for s in sessions:
                Xc = np.empty_like(sessions[s]["X"])
                for i in range(sessions[s]["X"].shape[0]):
                    Xc[i] = asr_process(sessions[s]["X"][i], sfreq=fs, M=M, T=Tcut)
                sessions[s]["X"] = Xc
        for s in sessions:
            sessions[s]["X"] = zscore_trials(sessions[s]["X"])

        feat_df, acc, metrics, regime, csp, sigma = analyse_subject(
            subject, sessions, info)

        all_acc[subject]     = acc
        all_metrics[subject] = metrics
        regimes[subject]     = regime

        # Per-subject summary figure
        plot_subject_summary(subject, feat_df, acc, metrics, regime, fig_dir)

        # Cache
        with open(os.path.join(fig_dir, f"{_safe(subject)}_cache.pkl"),"wb") as f:
            pickle.dump({"acc": acc, "metrics": metrics, "regime": regime}, f)
        with open(done_flag, "w") as f:
            f.write("done\n")

        del sessions, feat_df

    # -- Cohort-level figures -----------------------------------
    if all_acc:
        plot_accuracy_cohort(all_acc, regimes,
                             {s: list(all_acc[s].keys()) for s in all_acc},
                             fig_dir)
        plot_regime_summary_bar(regimes, fig_dir)
        plot_metric_trajectories_cohort(all_metrics, all_acc, regimes, fig_dir)

        # Results table
        df_results = build_results_table(all_acc, all_metrics, regimes)
        csv_path   = os.path.join(fig_dir, "results_table.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"\n[TABLE] saved to {csv_path}")
        print(df_results.to_string(index=False))

    print(f"\n[DONE] all figures in {fig_dir}")