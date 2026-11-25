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
| `B_decoding/B10_trial_level_master_table.py` | Trial-level master table of the slope and sinusoid measures | Only C1, C4a and C4b read it |
| `C_statistics/C1_replay_statistics.py` | Replay statistics on the master table, full and trial-matched | Earlier form of the sequentiality tests; the reported tests per time point are in C11 |
| `C_statistics/C2_item_reactivation.py` | Item-level reactivation from the B8 tables | Superseded by C9 (item time courses with matched control) |
| `C_statistics/C3_window_comparison.py` | Comparison of windows and versions of the measures | Decided: window 4.5-12 s, thesis preset of the sinusoid |
| `C_statistics/C4a_figure_panels_main.py`, `C4b_figure_panels_statistics.py` | Figure panels of the thesis chapter | Figures are now made by B5, B6b, C9, C10, C11, C12 |
| `C_statistics/C5_predecessor_reactivation.py` | Items before and after the retrieved one, by lag | Superseded by C9 |
| `C_statistics/C6_sequence_vs_control.py` | Retrieved against control sequence per session | Superseded by C11 |
| `C_statistics/C7_brain_behaviour.py` | Relation of the measures to reported retrieval | Superseded by C11 (both measures, reported retrieval and search time) |

The decoding scripts (B10, C1-C7) were moved on 2026-11-25 after the review of the
decoding analyses. C8 stays in the pipeline: it holds the test of the true order
against all orderings of the same probabilities, which C11 does not replace.
