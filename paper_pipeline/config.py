# -*- coding: utf-8 -*-
"""
Central configuration for the Sleeplay paper pipeline.

Every script in this folder takes its paths and constants from here, so the
pipeline can be moved to another machine by setting three environment
variables instead of editing scripts:

    SLEEPLAY_ROOT         project folder that contains `behavior/`
                          (default: nearest parent holding `.my_hidden_root_file`)
    SLEEPLAY_DERIVATIVES  fMRIPrep derivatives folder
    SLEEPLAY_OUTPUT       where all pipeline outputs are written
                          (default: <this folder>/outputs)
    SLEEPLAY_PLOTS        where all plots are written (optional)
                          (default: `plots` next to the output folder)

Rules the scripts follow:
  - raw data (behavior/, derivatives) is read-only; nothing is written there
  - each script writes to OUTPUT_ROOT/<script name>/ via step_output_dir()
  - each script saves its plots to PLOTS_ROOT/<script name>/ via step_plot_dir()
  - a script reads earlier results only from OUTPUT_ROOT/<earlier script name>/
"""

import os
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent


def _find_project_root() -> Path:
    env = os.environ.get("SLEEPLAY_ROOT")
    if env:
        return Path(env).resolve()
    for parent in PIPELINE_DIR.parents:
        if (parent / ".my_hidden_root_file").exists():
            return parent
    raise RuntimeError(
        "Project root not found. Set SLEEPLAY_ROOT to the folder that contains `behavior/`."
    )


PROJECT_ROOT = _find_project_root()
BEHAVIOR_DIR = PROJECT_ROOT / "behavior"
DERIVATIVES_DIR = Path(
    os.environ.get(
        "SLEEPLAY_DERIVATIVES",
        "/beegfs/g/schuck/bbd7515/sleeplay/l_loc_main/data/derivatives",
    )
)
OUTPUT_ROOT = Path(os.environ.get("SLEEPLAY_OUTPUT", PIPELINE_DIR / "outputs")).resolve()
# Plots go to a folder next to the outputs: <...>/plots/<script name>/
PLOTS_ROOT = Path(os.environ.get("SLEEPLAY_PLOTS", OUTPUT_ROOT.parent / "plots")).resolve()

# ── Sessions with a known recording problem ──────────────────────────────────
# Keys are (participant, session) with session "presleep" (evening) or
# "postsleep" (morning; both morning logs, nses_1 and nses_2).
#
# RELABELLED_SESSIONS: the retrieval targets written in the log are replaced by
# the participant's own learned item at the cued scene and position
# (decoding_utils.relabel_targets_from_cue). A1 applies it when it writes the
# behaviour tables and keeps the logged targets in `obj_ret_targ_logged` /
# `sc_ret_targ_logged`; `load_main_task_behavior` (B7) applies it again when the
# behaviour is loaded, which changes nothing in a table A1 already corrected.
#   s41 postsleep (decided with the user on 2026-11-25):
#       What happened. s41 learned on the evening of 2025-07-09 and was scanned
#       again on the morning of 2025-07-10 (dates of the raw log files: evening
#       22:11; morning 08:28 and 08:59). That morning the task was started with
#       participant number 42 instead of 41 (Data_overview.xlsx: "wrong number
#       entered during recording - renamed to 41"). The task builds its trial
#       list from the participant number, so s41 was asked participant 42's
#       list of questions. s42 is a different person (evening 2025-07-10,
#       morning 2025-07-11).
#       How it was found. In the morning logs of s41 only 6 of 100 object
#       targets are objects s41 learned at the cued scene and position; all 100
#       match what s42 learned. The trial order is identical to that of s42's
#       own morning session, the logs store number 42 (`sub_idx` 41, zero-based),
#       and the responses differ from s42's, so it is a separate recording.
#       Why it can be used. A question shows a scene and a position; all
#       participants learned the same ten scenes, so the question is valid for
#       s41, who answers with the object s41 learned there. Only the "correct
#       answer" in the log belongs to s42. Every (scene, position) occurs once per
#       pass, so after the correction all 50 of s41's objects are asked once per
#       pass, in the order of s42's list. The rule "the object learned at the
#       cued position of the cued scene" (scene trials: the scene learned at the
#       cued position of the cued context) reproduces the logged target in every
#       other session of every participant. With the rebuilt targets the
#       classifier evidence for the retrieved item in s41's visual cortex is
#       +2.3 points (SE 1.7); with the logged targets it is -3.7 (SE 1.6).
#
# EXCLUDED_SESSIONS: sessions that must not enter any analysis that depends on
# which item a retrieval trial asks for. Empty at present; B8 (sequence
# validation) and D1 `retrieval_events` (hierarchy check) drop any session whose
# targets do not match the learned sequences.
RELABELLED_SESSIONS = {
    ("s41", "postsleep"): "task started with participant number 42; targets rebuilt from the cue and s41's learned sequences",
}
EXCLUDED_SESSIONS: dict = {}


def _sub_key(subject: str) -> str:
    return "s" + str(subject).replace("sub-", "").lstrip("s")


def is_excluded(subject: str, session: str) -> bool:
    """True if this participant's session is excluded (subject as 's41' or 'sub-41')."""
    return (_sub_key(subject), session) in EXCLUDED_SESSIONS


def is_relabelled(subject: str, session: str) -> bool:
    """True if the retrieval targets of this session are rebuilt from the cue."""
    return (_sub_key(subject), session) in RELABELLED_SESSIONS


# ── Acquisition constants ────────────────────────────────────────────────────
TR = 1.5          # seconds
HRF_DELAY = 5     # seconds added to event onsets before converting to TRs

# ── Sessions ─────────────────────────────────────────────────────────────────
SESSION_LABELS = {0: "Evening (S0)", 1: "Morning 1 (S1)", 2: "Morning 2 (S2)"}


def step_output_dir(script_file, subfolder: str | None = None) -> Path:
    """Output folder of a pipeline step: OUTPUT_ROOT/<script name>[/subfolder]."""
    out = OUTPUT_ROOT / Path(script_file).stem
    if subfolder:
        out = out / subfolder
    out.mkdir(parents=True, exist_ok=True)
    return out


def step_plot_dir(script_file, subfolder: str | None = None) -> Path:
    """Plot folder of a pipeline step: PLOTS_ROOT/<script name>[/subfolder].

    Every figure a step draws goes here, so the folder name says which script
    made it. Tables and other data stay in step_output_dir().
    """
    out = PLOTS_ROOT / Path(script_file).stem
    if subfolder:
        out = out / subfolder
    out.mkdir(parents=True, exist_ok=True)
    return out


def step_input_dir(step_name: str, subfolder: str | None = None) -> Path:
    """Output folder of an earlier step, e.g. step_input_dir('A1_add_tr_columns_main')."""
    src = OUTPUT_ROOT / step_name
    if subfolder:
        src = src / subfolder
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist. Run {step_name}.py first.")
    return src


def behavior_subjects() -> list[str]:
    """Subject folders in behavior/, e.g. ['s01', 's11', ...]."""
    return sorted(
        p.name for p in BEHAVIOR_DIR.iterdir() if p.is_dir() and p.name.startswith("s")
    )


def find_closest_TR(target, start=0, end=1000, tr=TR):
    """Index of the TR whose onset is closest to `target` seconds.

    Kept exactly as in the original preprocessing scripts so that the TR
    columns are reproduced bit for bit (ties go to the earlier TR; a NaN
    target returns None).
    """
    numbers = []
    while start <= end:
        numbers.append(start)
        start += tr
    closest_index = None
    closest_difference = float("inf")
    for i, num in enumerate(numbers):
        difference = abs(num - target)
        if difference < closest_difference:
            closest_index = i
            closest_difference = difference
    return closest_index
