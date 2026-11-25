# -*- coding: utf-8 -*-
"""Stage 2: combinations of the factors that helped in stage 1.

Stage 1 (35 participants, one factor at a time): stronger regularisation
(logistic regression C = 0.001, or the correlation classifier) and training on
shown trials only raised the accuracy; no feature selection was neutral; all
other changes were neutral or worse. These are combined here. `mtl_current` is
repeated so that the stage can be summarised on its own.
"""


def build(CURRENT, _v):
    hpc = dict(CURRENT, roi="hpc")
    v = [_v("mtl_current", CURRENT)]
    for clf_name, clf_kw in (("corr", dict(clf="corr")), ("lrC0.001", dict(C=0.001)),
                             ("lrC0.0001", dict(C=0.0001))):
        v.append(_v(f"mtl_{clf_name}_fs100", CURRENT, fs=100, **clf_kw))
        v.append(_v(f"mtl_{clf_name}_trainshown", CURRENT, train="shown", **clf_kw))
        v.append(_v(f"mtl_{clf_name}_fs100_trainshown", CURRENT, fs=100, train="shown", **clf_kw))
        v.append(_v(f"mtl_{clf_name}_trainrecalled", CURRENT, train="recalled", **clf_kw))
        v.append(_v(f"hpc_{clf_name}_fs100", hpc, fs=100, **clf_kw))
    v.append(_v("mtl_lrC0.0001", CURRENT, C=0.0001))
    # voxel sets and window with the correlation classifier
    v.append(_v("mtl_corr_rel10", CURRENT, clf="corr", rel=10))
    v.append(_v("mtl_corr_anat", CURRENT, clf="corr", rel=None))
    v.append(_v("phc_corr_rel70", CURRENT, clf="corr", roi="phc"))
    v.append(_v("phc_corr_anat_fs100", CURRENT, clf="corr", roi="phc", rel=None, fs=100))
    v.append(_v("mtl_corr_win3tr", CURRENT, clf="corr", win=(-1, 0, 1)))
    v.append(_v("mtl_corr_smooth2", CURRENT, clf="corr", smooth=2))
    v.append(_v("hpc_corr_fs100_trainshown", hpc, clf="corr", fs=100, train="shown"))
    return v
