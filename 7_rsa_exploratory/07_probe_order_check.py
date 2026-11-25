#!/usr/bin/env python3
"""
Is sequence-position distance related to the order in which objects were probed at retrieval?

For object trials of the same scene that lie in different runs (the pairs the RSA
uses), compare how far apart in probe order direct neighbours (position distance 1)
and more distant objects (distance >= 2) were tested. Uses only the trial tables.

Usage:  python 07_probe_order_check.py
Output: outputs/07_probe_order_check/probe_order_by_subject.csv, probe_order_summary.csv
"""

import glob
import os
from os.path import join as opj

import numpy as np
import pandas as pd
from scipy import stats

import rsa_config as cfg

OUT_DIR = opj(cfg.OUT_ROOT, "07_probe_order_check")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for session in cfg.SESSIONS:
        for f in sorted(glob.glob(opj(cfg.OUT_ROOT, "patterns_lss", session, "*_trials.csv"))):
            t = pd.read_csv(f)
            t = t[t.kind == "object"].sort_values(["run_idx", "onset_s"]).reset_index(drop=True)
            parts = [("evening", t)] if session == "presleep" else \
                    [("morning pass 1", t[t.run_idx < 3]), ("morning pass 2", t[t.run_idx >= 3])]
            for name, g in parts:
                g = g.reset_index(drop=True)
                i, j = np.triu_indices(len(g), 1)
                dpos = np.abs(g.position.values[i] - g.position.values[j])
                dord = np.abs(i - j)                                   # trials apart in probe order
                between = g.run_idx.values[i] != g.run_idx.values[j]
                m = (g.scene.values[i] == g.scene.values[j]) & between
                use = between & (dpos > 0)
                rows.append(dict(subject=os.path.basename(f).split("_")[0], set=name,
                                 n_pairs=int(m.sum()),
                                 trials_apart_neighbours=dord[m & (dpos == 1)].mean(),
                                 trials_apart_distant=dord[m & (dpos >= 2)].mean(),
                                 r_position_vs_probe_distance=np.corrcoef(dpos[use], dord[use])[0, 1]))
    d = pd.DataFrame(rows)
    d.to_csv(opj(OUT_DIR, "probe_order_by_subject.csv"), index=False)
    out = []
    for name, g in d.groupby("set", sort=False):
        t, p = stats.ttest_rel(g.trials_apart_neighbours, g.trials_apart_distant, nan_policy="omit")
        out.append(dict(set=name, n=len(g), neighbours=g.trials_apart_neighbours.mean(),
                        distant=g.trials_apart_distant.mean(), t=t, p=p,
                        mean_r=g.r_position_vs_probe_distance.mean()))
    s = pd.DataFrame(out)
    s.to_csv(opj(OUT_DIR, "probe_order_summary.csv"), index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
