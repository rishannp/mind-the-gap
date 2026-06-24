# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
SHU MI - co-adaptive FBCSP-LDA, pseudo-online.
==============================================
The 5 SHU subjects with the largest jump-regime accuracy collapse. Each
has 5 real sessions; the stream concatenates them in session order and we
keep the session boundaries so they can be marked on the per-subject plots.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import coadapt_core as cc

# ============================================================
# CONFIG
# ============================================================
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Multi-Session Data\SHU Dataset\MatFiles'
fig_dir  = r'C:\Users\uceerjp\Desktop\PhD\THESIS\THESIS - Chapter 3\SHU\shu_coadapt_figures'
regime_csv = r'C:\Users\uceerjp\Desktop\PhD\THESIS\THESIS - Chapter 3\SHU\shu_nonstationarity_figures\shu_results_table.csv'

fs       = 250
SUBJECTS = ["sub-001", "sub-002", "sub-013", "sub-016", "sub-009"]


def load_subject(sid):
    """Concatenate the subject's sessions; return trials, labels, bounds.

    bounds = global trial index at which each session after the first starts.
    """
    files = sorted(f for f in os.listdir(data_dir)
                   if f.startswith(sid + "_") and f.endswith(".mat"))
    trials, labels, bounds = [], [], []
    for f in files:
        mat = loadmat(os.path.join(data_dir, f))
        X   = mat["data"]                          # [trials x ch x samples]
        y   = mat["labels"].flatten()
        if len(labels) > 0:
            bounds.append(len(labels))             # boundary before this session
        for t in range(X.shape[0]):
            trials.append(X[t].astype(np.float64))
            labels.append(int(y[t]))
    return trials, labels, bounds


def load_regimes():
    df = pd.read_csv(regime_csv)
    return dict(zip(df["Subject"], df["Regime"]))


if __name__ == "__main__":
    regimes = load_regimes()
    streams = {}
    for sid in SUBJECTS:
        trials, labels, bounds = load_subject(sid)
        streams[sid] = (trials, labels, bounds)
        print(f"[LOAD] {sid}: {len(labels)} trials "
              f"(L={labels.count(1)}, R={labels.count(2)}), "
              f"sessions start at {bounds}")

    cc.run_dataset("SHU", streams, regimes, fig_dir, fs)
