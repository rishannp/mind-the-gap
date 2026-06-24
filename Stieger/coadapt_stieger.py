# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Stieger MI - co-adaptive FBCSP-LDA, pseudo-online.
==================================================
The 5 Stieger subjects with the largest jump-regime accuracy collapse
(S43, S29, S19, S57, S39). Reads the raw S{n}_Session_{n}.pkl files,
keeps the motor-channel subset, concatenates sessions in order.

Set DATA_DIR to wherever the .pkl files live (the cluster scratch dir, or
a local copy of just these 5 subjects). coadapt_core.py must be importable
- keep it next to this file or on PYTHONPATH.
"""

import os
os.environ["MNE_BROWSER_BACKEND"] = "matplotlib"
os.environ["QT_QPA_PLATFORM"]     = "offscreen"

import re
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coadapt_core as cc

# ============================================================
# CONFIG  (runs on the cluster, like Stieger_CSP.py)
# ============================================================
server_dir = "/scratch/uceerjp"        # holds the S{n}_Session_{n}.pkl files
DATA_DIR   = server_dir
FIG_DIR    = os.path.join(server_dir, "stieger_coadapt_figures")

fs       = 1000
SUBJECTS = ["S43", "S29", "S19", "S57", "S39"]
REGIMES  = {s: "jump" for s in SUBJECTS}   # all selected on the jump criterion

# ============================================================
# ELECTRODES  (full montage; motor subset selected below)
# ============================================================
FULL_LABELS = [
    'FP1','FPZ','FP2','AF3','AF4',
    'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
    'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
    'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
    'P7','P5','P3','P1','PZ','P2','P4','P6','P8',
    'PO7','PO5','PO3','POZ','PO4','PO6','PO8',
    'CB1','O1','OZ','O2','CB2'
]


def motor_channels(labels):
    """Central/parietal motor strip; drop frontal-pole, temporal, occipital."""
    keep = []
    for i, ch in enumerate(labels):
        u = ch.upper()
        if u.startswith(("FP", "AF", "FT", "TP", "PO", "CB", "O")):
            continue
        if u.startswith(("F", "FC", "C", "CP", "P")):
            keep.append(i)
    return np.array(keep)


KEEP_IDX = motor_channels(FULL_LABELS)


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


def list_sessions(sid):
    pat = re.compile(rf"^{sid}_Session_(\d+)\.pkl$")
    found = []
    for f in os.listdir(DATA_DIR):
        m = pat.match(f)
        if m:
            found.append((int(m.group(1)), os.path.join(DATA_DIR, f)))
    return [p for _, p in sorted(found)]


def load_subject(sid):
    """Concatenate sessions; return trials [ch x samples], labels, bounds."""
    trials, labels, bounds = [], [], []
    for path in list_sessions(sid):
        with open(path, "rb") as fh:
            bci = pickle.load(fh)

        sess_trials, sess_labels = [], []
        for trial, meta in zip(bci["data"], bci["TrialData"]):
            lab = meta.get("targetnumber")
            if lab not in (1, 2):
                continue
            eeg = np.asarray(trial, dtype=np.float64)
            if eeg.shape[0] != len(FULL_LABELS):
                eeg = eeg.T
            if eeg.shape[0] != len(FULL_LABELS):
                continue
            sess_trials.append(eeg[KEEP_IDX, :])
            sess_labels.append(int(lab))

        if not sess_trials:
            continue
        # equalise length within session (modal), as in the CSP script
        target = modal_len([t.shape[1] for t in sess_trials])
        sess_trials = [pad_or_trim(t, target) for t in sess_trials]

        if len(labels) > 0:
            bounds.append(len(labels))
        trials.extend(sess_trials)
        labels.extend(sess_labels)
    return trials, labels, bounds


if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"DATA_DIR not found: {DATA_DIR}\n"
                         f"Point it at the folder holding "
                         f"{', '.join(s + '_Session_*.pkl' for s in SUBJECTS)}")

    streams = {}
    for sid in SUBJECTS:
        trials, labels, bounds = load_subject(sid)
        if not trials:
            print(f"[MISS] {sid}: no .pkl files found in {DATA_DIR}")
            continue
        streams[sid] = (trials, labels, bounds)
        print(f"[LOAD] {sid}: {len(labels)} trials "
              f"(L={labels.count(1)}, R={labels.count(2)}), "
              f"sessions start at {bounds}")

    cc.run_dataset("Stieger", streams, REGIMES, FIG_DIR, fs)
