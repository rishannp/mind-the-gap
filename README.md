# Mind the Gap

Quantifying distributional jumps in longitudinal motor-imagery EEG, and testing whether co-adaptation can actually close the session-boundary gap.

---

## What this is

This repository holds the code for a thesis chapter on **why brain-computer interfaces (BCIs) stop working over time**, and whether the usual fix really works.

A motor-imagery BCI lets a person control a computer by imagining left- or right-hand movement. The catch is well known: a model trained today often performs much worse next week, even on the same person. The standard explanation is "non-stationarity", the brain signals drift. The standard fix is "co-adaptation", keep retraining the model as new data arrives.

This work asks two plain questions about that story:

1. **When the signal changes between sessions, is it a smooth drift or a sudden jump?**
2. **If it is a jump, does retraining the model actually fix it?**

We answer both with the same simple trick: freeze part of the system, then watch what happens to everything else.

---

## The two experiments

### Experiment 1: Do distributional jumps exist?

We take a model that has been trained once and never updated, and we watch how the data moves underneath it.

1. Train a spatial filter (CSP) on the first session only, then **freeze it**.
2. Push every later session through that frozen filter and look at where the features land.
3. Measure three things over time:
   - **Accuracy**: how well a frozen classifier still labels left vs right.
   - **Within-session separability**: at each session, are the two classes still distinguishable from each other? (measured with symmetric KL divergence and MMD)
   - **Between-session drift**: how far has the overall feature distribution moved away from session one? (measured with MMD, with a permutation test for significance)

If the world only drifted gently, accuracy would fade slowly and the classes would stay separable. What we actually see for many subjects is different: accuracy falls to chance after the first session and the two classes stop being separable at all. That is the signature of a **jump**, not a drift. The point is not just that the data moved, but that the structure the model relied on fell apart.

We do this for every subject and sort them into two groups: **jump** (sudden collapse) and **ambiguous** (everything else, usually subjects who were near chance from the start).

### Experiment 2: Can co-adaptation fix it?

Now we let a model move and see if it can keep up.

1. Train a filter-bank CSP plus a simple classifier (LDA) on a small starting set of 60 trials.
2. Keep a **frozen copy** of that model that never changes. This is the control.
3. Replay each subject's trials in time order, one at a time. For every incoming trial:
   - the live model predicts it first (this is the score we record),
   - then the trial is added to a buffer with its true label.
4. Once the buffer holds 30 fresh trials, **retrain the spatial filters and the classifier on just those latest trials**, throw the buffer away, and carry on. Old data is discarded so the model always reflects the current distribution.

This is "test before you train", which is the honest way to measure an online system. The model is always judged on data it has not seen yet.

The headline number is **recovery**: how much better the adapting model does than its frozen twin on the same trials. Positive recovery means adaptation helped. Near zero means it did not.

---

## What we found, in short

- **Jumps are real and they show up in all three datasets.** The feature distributions move significantly between sessions for almost everyone, and a clear group of subjects suffer a sudden collapse in both accuracy and class separability.
- **Co-adaptation helps, but only modestly, and mostly for the wrong group.** Subjects with gentle, continuous drift recover well. Subjects with genuine jumps recover much less, and some never get back above chance no matter how much the model retrains.
- **The takeaway:** retraining the classifier and the spatial filters cannot rebuild structure that has already fallen apart. When a real jump happens, the problem is the feature representation itself, not the choice of classifier or how often you retrain it. This motivates the next chapter, which looks for a representation that stays stable across the gap rather than a better way to chase an unstable one.

---

## The datasets

| Dataset | Population | Subjects used | Sessions | Channels | Sample rate |
|---|---|---|---|---|---|
| ALS | Motor-impaired (ALS) | all 8 | causal folds of one recording | 19 | 256 Hz |
| SHU | Healthy | 5 (worst jumps) | 5 real sessions | 32 | 250 Hz |
| Stieger | Healthy | 5 (worst jumps) | up to 10 real sessions | 39 (motor subset) | 1000 Hz |

For the co-adaptation experiment we keep all 8 ALS subjects (so we get both jump and non-jump subjects in one group), and from the two larger healthy cohorts we pick the 5 subjects with the worst jumps, measured as the largest drop in frozen accuracy. The ALS group gives the within-dataset contrast; the SHU and Stieger groups test whether the worst jumps resist recovery across different people and protocols.

---

## How the code is organised

```
THESIS - Chapter 3/
  coadapt_core.py          shared co-adaptation engine (FBCSP, online loop, plots, tables)
  make_results_latex.py    builds the LaTeX figures and the combined results table from the CSVs

  ALS/
    CSP-ALS.py             Experiment 1 (non-stationarity characterisation)
    coadapt_als.py         Experiment 2 runner
    als_nonstationarity_figures/   outputs from Experiment 1
    als_coadapt_figures/           outputs from Experiment 2

  SHU/
    CSP-Healthy.py         Experiment 1
    coadapt_shu.py         Experiment 2 runner
    ...figures folders

  Stieger/
    Stieger_CSP.py         Experiment 1 (runs on the cluster)
    coadapt_stieger.py     Experiment 2 runner (runs on the cluster)
    ...figures folders
```

The characterisation scripts (`CSP-*.py`, `Stieger_CSP.py`) are standalone. The co-adaptation runners are thin: they only load each dataset and hand a clean trial stream to `coadapt_core.py`, which does the actual work. That keeps the method in one place so all three datasets are treated identically.

---

## Running it

Each script sets its own data path at the top. ALS and SHU run locally; Stieger reads from the cluster scratch directory (`/scratch/uceerjp`), so `coadapt_stieger.py` and `coadapt_core.py` need to sit together on the cluster.

```bash
# Experiment 1 (per dataset)
python ALS/CSP-ALS.py
python SHU/CSP-Healthy.py
python Stieger/Stieger_CSP.py          # on the cluster

# Experiment 2 (per dataset)
python ALS/coadapt_als.py
python SHU/coadapt_shu.py
python Stieger/coadapt_stieger.py      # on the cluster

# Build the LaTeX figures and combined table from the saved CSVs
python make_results_latex.py
```

Outputs for each dataset land in its own `*_figures` folder: per-subject plots, a cohort plot, per-block and summary CSVs, and a subjects-by-block accuracy table.

Dependencies: `numpy`, `scipy`, `scikit-learn`, `pandas`, `mne`, `matplotlib`.

---

## A few honest caveats
- The co-adaptation uses true labels to retrain, so it is an upper bound on what is achievable, not a ready-to-deploy estimate.
- "Jump" versus "ambiguous" is a labelling convenience on top of the underlying measurements; the measurements themselves (drift significance, separability collapse) are the real evidence.
