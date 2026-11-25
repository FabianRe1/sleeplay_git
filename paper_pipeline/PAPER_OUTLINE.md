# Paper outline (draft, 2026-11-19)

Working title: **Sequential reactivation during retrieval extends beyond the
retrieved item: stable across a night of sleep, with a region-specific shift at
the first retrieval after sleep.**

Numbers are from the existing results (before the full pipeline run) and are
uncorrected unless stated. "Retrieval success" is the term used throughout; it is
measured as the participant pressing to indicate retrieval (85 % of trials). The
spoken answers are not scored yet, so the methods must say how it is defined.
Terms decided on 2026-11-19: "sequentiality" for the slope and sinusoid measures,
"retrieval success" for the behavioural measure, the caching idea in the
discussion only.
Three time points are kept apart throughout: evening, first morning retrieval,
repeated morning retrieval.

Order of the neural results (decided 2026-11-22): mean activity first, then all
decoding results, then pattern similarity. Each block reports its own comparison
of time points; section 5 is the synthesis for the start of the discussion.

## 1. Behaviour: an ordered search that speeds up without changing character

- Reaction time rises with sequence position in every session (object trials,
  cost per position 16-17 % of mean reaction time at all three time points).
- The last position is answered faster than the fourth (-0.3 to -0.5 s in every
  session): the last item is reached by another route.
- Reaction times fall by 0.5 s overnight and by a further 0.4 s at the repeated
  retrieval; the position cost falls in proportion.
- Wording: "retrieval time depends on the item's position in the sequence",
  "the search speeds up as a whole". The caching idea is not part of the
  results (see discussion).
- Steps: A5, A6.

## 2. Mean activity: a stable retrieval network that follows retrieval time

Order within the section: overall profile, then per time point, then the
relation to reaction time (decided 2026-11-22: the basic maps come before the
precuneus result).

**2a. Which regions are engaged (whole brain, TFCE-corrected).**
- Above baseline at retrieval: visual cortex, posterior precuneus and
  parieto-occipital cortex (peak t = 12.9), medial frontal cortex (12.5),
  caudate, putamen, thalamus, parahippocampal cortex, cerebellum.
- At baseline: hippocampus, inferior parietal cortex. Below baseline: lateral
  temporal, medial prefrontal, sensorimotor cortex.
- The precuneus is split: posterior part engaged, anterior dorsal part below
  baseline. Locate the result as posterior precuneus / parieto-occipital cortex.
- Wording: the flat hippocampus reflects the contrast against baseline, not
  absence of involvement.

**2b. The same at each time point; changes between time points.**
- The layout is the same in the evening, at the first and at the repeated
  morning retrieval.
- Evening to first morning: only caudal anterior cingulate rises under both
  event models (t = 3.8 / 3.6, corrected). No increase in precuneus or parietal
  cortex (t = 0.5 to 1.7), no decrease in hippocampus (0.4 / 1.5).
- First to repeated morning: hippocampus (-2.5 / -2.7), parahippocampal cortex
  (-5.6 / -3.2), MTL (-4.3 / -3.8) and caudal anterior cingulate (-4.2 / -3.4)
  drop; parietal + precuneus stays level. Rest this on the region analysis in
  native space; at the whole-brain level it passes correction with fixed events
  only (left putamen rises under both).
- Report both event models; a difference counts only if it holds under both.

**2c. Engaged for as long as retrieval takes.**
- Activity rises with reaction time in posterior precuneus (whole brain
  t = 7.6; region t = 7.0, at every time point), retrosplenial and visual
  cortex, medial frontal cortex, caudate, anterior insula; not in hippocampus.
- With trials modelled from cue to response the relation disappears in
  precuneus (t = -0.4). Wording: "consistent with sustained engagement";
  stronger activity on slow trials in exact proportion to their duration
  cannot be excluded.

**Main text / supplement.** Main: pooled map, region table of changes under both
event models, reaction-time map. Supplement: per-time-point maps, maps with
events from cue to response, whole-brain difference maps, the simulation that
explains the two event models.

**Checked and not carried further.** Restricting to trials with a button press
gives the same maps (r = .99). A decoding region defined from the engaged
posterior precuneus shows no decoding effect (classifier 12 %, retrieved item
t = 0.4): engagement and item content are in different places, so the
anatomical parietal + precuneus region stays the decoding region.

- Steps: D1, E1, E2 (regions, native space), E3, E4, E5 (whole brain).

## 3. Decoding: the retrieved item, its neighbouring positions, and sequentiality

- The retrieved item is reinstated without any picture on screen in visual
  cortex (t = 4.7), parietal + precuneus (3.9) and MTL (2.3); not in the control
  region.
- The item before it is reactivated as well, in visual cortex (t = 5.2) and
  parietal + precuneus (4.5). Items two or more steps away are not.
- **Sequentiality is not only driven by the retrieved item.** The sequentiality
  measures (slope, sinusoid amplitude) hold when the retrieval target is left
  out of the sequence; the evidence then sits on the neighbouring positions
  (the item before the target, and at the first morning retrieval in
  parietal + precuneus also the item after it).
- Stronger reinstatement of the retrieved item and stronger sequentiality go
  with higher retrieval success between participants (retrieved item: visual
  r = .51, parietal + precuneus r = .43; sequentiality in parietal + precuneus
  r = .42-.57); evidence for the neighbours alone does not.
- To state in the methods or as a limit: the sequentiality measures respond to
  items of the sequence standing out next to the target. In a control that
  compares the true order with all other orderings of the same evidence, the
  true order is not favoured (C8), so the measures index reactivation of the
  target's neighbourhood within the sequence, not a graded sweep across all
  five positions. A reviewer may ask for this control.
- Further limits: with trials restricted to successful retrieval the
  predecessor remains but the difference between predecessor and successor
  does not; scene trials show no neighbour effects.
- Steps: B5, C2, C5, C7, C8.

## 4. A graded position code in pattern similarity

- Objects at neighbouring positions have more similar activity patterns than
  objects at distant positions, in visual, parietal, precuneus and frontal
  regions (t = 4-6), at all three time points, across scenes, contexts and
  categories.
- Report it with the response-time controls (D3 part E, added 2026-11-23):
  response time differs by position and shapes similarity in every region.
  With the response-time difference of the two trials and the mean response
  time per position in the model, the effect is t = 2.4 to 3.4 (visual,
  parietal, frontal; FDR p = .014 to .045) and gone in retrosplenial cortex.
- The binary same-position effect is a digit effect (it appears between object
  and scene trials, where the digit means different things) and is not
  reported as a position code.
- Stated limits: the cue shows the position digit, so a digit or order code is
  not excluded; the scene effect is explained by the scene picture in the cue;
  no context, scene-order or session effects.
- Mention only as a lead: neighbours within the same scene are additionally
  similar (depends on the estimator; not specific to one session).
- Steps: D1-D4.

## 5. Three time points, not two sessions (synthesis; decoding part belongs to section 3, mean activity to section 2)

**Stable across the night:** the position code, reinstatement of the retrieved
item, reactivation of the predecessor, and mean activity in the regions of
interest.

**At the first retrieval after sleep, in parietal + precuneus only:**
- sequentiality (sinusoid amplitude) appears (evening t = -0.2, first morning
  retrieval t = 3.8; change p = .014);
- it is not only driven by the retrieved item: it holds without the retrieval
  target (change p = .030), and more than in visual cortex (p = .045) and MTL
  (p = .047);
- without the target the evidence has shifted to the neighbouring positions:
  the item after the retrieved one gains evidence (evening t = -0.6, first
  morning retrieval t = 2.3; change p = .043), while the item before it is
  present at all time points.
- Wording: "at the first retrieval after sleep, sequentiality in
  parietal + precuneus increases; it is not only driven by the retrieved item
  but holds without the retrieval target, where it has shifted to the
  neighbouring positions". Avoid attributing the change to sleep as such
  (no wake control).

**At the repeated morning retrieval:** that effect is gone; activity in
hippocampus, parahippocampal cortex and MTL drops (t = -2.4 to -5.6); the
predecessor is reactivated again.
- Wording: "the two morning retrievals differ and cannot be pooled".

- Steps: C5, C6, C8, E1.

## 6. Open questions and limits (for the discussion)

- No wake control: the overnight change cannot be attributed to sleep rather
  than time or re-testing.
- The overnight tests sit at p = .01-.05 before correction; they need one
  joint model (lag x time point x region) and a shuffled-sequence baseline for
  the amplitude without the retrieved item.
- The overnight change does not relate to retrieval success.
- Caching (discussion only): the repeated morning retrieval is faster, with
  lower activity in hippocampus and parahippocampal cortex and without the
  sequentiality increase, which could suggest direct access to an answer given
  shortly before. Against it: the position cost in reaction time stays the same
  share of retrieval time, and the item before the target is reactivated again.
- MTL: evidence for the following item (t = 3.1), but the classifier is close
  to chance there, the signal comes from parahippocampal cortex, the hippocampus
  itself decodes at chance, and pattern similarity shows no sequence structure
  in MTL. Present as a lead; no claim about a hippocampal predictive map.
- The position code is uneven (positions 1-2 and 3-4 are similar, 2-3 and 4-5
  are not); unexplained.
- Retrieval success is the participant's own report. Spoken answers are not
  scored yet; with them, accuracy and the kind of errors (neighbouring item or
  not) can be tested.

## Not used in the outline (available in the pipeline)

Detection of the five-item sequence measure in three regions with shuffled
baseline (C1), early versus late trials within the morning (C1), fit quality of
the sinusoid (C1), full versus truncated windows (C3), object versus scene
trials (C1, C2), encoding-session decoding as a check (B6), evidence ahead and
behind with different horizons (C5).
