# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
ALS MI - co-adaptive FBCSP-LDA, pseudo-online.
==============================================
All 8 ALS subjects. Each subject is a single continuous recording, so the
'stream' is reconstructed by interleaving the stored L and R trials (their
true interleaving is not preserved in the .mat layout - see thesis note).

Regimes are read back from the non-stationarity analysis so the recovery
figures can be split jump vs ambiguous.
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
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Multi-Session Data\OG_Full_Data'
fig_dir  = r'C:\Users\uceerjp\Desktop\PhD\THESIS\THESIS - Chapter 3\ALS\als_coadapt_figures'
regime_csv = r'C:\Users\uceerjp\Desktop\PhD\THESIS\THESIS - Chapter 3\ALS\als_nonstationarity_figures\als_results_table.csv'

fs          = 256
N_CHANNELS  = 19
SUBJECT_IDS = [1, 2, 5, 9, 21, 31, 34, 39]


def remove_trailing_zeros(lst):
    if lst and np.all(lst[-1] == 0):
        return lst[:-1]
    return lst


def load_subject(num):
    """Return (L_trials, R_trials), each a list of [n_ch x n_samples]."""
    mat = loadmat(os.path.join(data_dir, f"S{num}.mat"))
    arr = mat[f"Subject{num}"][0]
    L = remove_trailing_zeros([it["L"] for it in arr])
    R = remove_trailing_zeros([it["R"] for it in arr])
    # stored as [samples x channels]; keep first 19 ch, transpose to [ch x samples]
    L = [t[:, :N_CHANNELS].T for t in L]
    R = [t[:, :N_CHANNELS].T for t in R]
    return L, R


def build_stream(L, R):
    """Interleave L/R into one chronological-ish trial stream."""
    trials, labels = [], []
    for i in range(max(len(L), len(R))):
        if i < len(L):
            trials.append(L[i]); labels.append(1)
        if i < len(R):
            trials.append(R[i]); labels.append(2)
    return trials, labels


def load_regimes():
    df = pd.read_csv(regime_csv)
    return dict(zip(df["Subject"], df["Regime"]))


if __name__ == "__main__":
    regimes = load_regimes()
    streams = {}
    for num in SUBJECT_IDS:
        L, R = load_subject(num)
        trials, labels = build_stream(L, R)
        # ALS is single-session: no session boundaries
        streams[f"S{num}"] = (trials, labels, [])
        print(f"[LOAD] S{num}: {len(labels)} trials "
              f"(L={labels.count(1)}, R={labels.count(2)})")

    cc.run_dataset("ALS", streams, regimes, fig_dir, fs)
