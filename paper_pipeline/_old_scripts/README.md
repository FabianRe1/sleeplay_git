# Scripts no longer part of the pipeline

Moved here on 2026-11-25 to keep the pipeline folders to the steps that produce
reported results. Nothing was deleted; outputs of these steps are untouched.
To use one again, move it back into its stage folder (the scripts find `config.py`
and the other steps relative to that folder).

| Script | What it did | Why it is not used |
| --- | --- | --- |
| `E_univariate/E1_session_comparison.py` | Item-matched session differences from the single-trial estimates of D1, for events of a fixed 3 s and events from cue to response | Superseded by the main model decided on 2026-11-23 (E7) and the comparison matched for search time (E10) |
| `E_univariate/E2_figures.py` | Figures of E1 | Superseded with E1; the region figures are now made by E14 |
| `E_univariate/E6_event_model_illustration.py` | Simulation of what the two event models of E1 estimate | Illustrates the two-model approach that was dropped |
| `E_univariate/E8_native_then_standard.py` | Check: fit in native space, then transform the maps to standard space | A verification that was done once; `E4_MAPS=native_first` in E4 still reads its outputs if they exist |
| `E_univariate/E11_subsequent_memory.py` | Activity while learning, split by later retrieval | Null result, not reported |
