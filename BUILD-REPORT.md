# GTA VI Twitch CCU forecast — BUILD REPORT (v2)

## Entry
- Competition: `gta6-twitch-ccu-forecast-01` — forecast GTA VI's launch-day
  Twitch viewers, hours 2–24 (UTC), five quantiles an hour.
- Repository: `~/workspace/makefun/gta6-twitch-ccu-forecast-01-entry-v2/`
  (v1 lives untouched in `...-entry/`).
- Entrypoint: `main.py` (Python 3.13, pure stdlib, single file).
- Built: 2026-11-17 ~18:30–19:00 WIB, by one subagent session.
- Status: built + verified locally. NOT submitted — the parent submits only
  if v2 clearly beats v1.

## Why v2 (what was wrong with v1)
v1 (practice mean 0.1495) under-predicts the evening and the morning rise
and over-predicts the early hours and the midday dip. Its per-hour median
log-errors on practice show a systematic shape gap:

| hours | v1 bias (log-err, actual − pred) |
|-------|----------------------------------|
| 2–4   | −0.12 … −0.19 (over-predicts early) |
| 5–7   | +0.24 … +0.33 (under-predicts morning rise) |
| 12–14 | −0.17 … −0.33 (over-predicts midday) |
| 18–21 | +0.12 … +0.24 (under-predicts evening) |

The real launch-day curves have MORE diurnal amplitude than the
comparables' smeared average (the 38 comparables mix rolling-midnight,
simultaneous and other unlocks, so their per-hour average is flat).
Diagnosis: when an all-comparables OLS model predicts ROLLING
comparables (comparable-LOO), residuals are +0.22…+0.36 at hours 6–24.
v1 only half-corrects this, because only its ridge component (weight 0.5)
carries the rolling indicator; the OLS-2feat and kernel components
(weight 0.5) dilute it.

v2 gives the rolling regime more weight twice over:
1. **Kernel roll boost (×3)** in the base ensemble: rolling comparables'
   kernel weights are tripled.
2. **A dedicated rolling-subset ensemble blended in at 25%**: the v1
   ensemble (ridge3 0.5 / OLS-2feat 0.2 / kernel 0.3) fit on the
   rolling-midnight comparables only.

The base ensemble drops the OLS-2feat component (ridge3 0.5 +
kernel-roll×3 0.5); the rolling component keeps v1's weights.

## Approach
The count at hour h is predicted as a log-ratio to hour 1's print:
`ln(viewers_h) = ln(p1) + r_h`. Final median = 0.75 × base median +
0.25 × rolling median; spread likewise. Spread is the robust (MAD)
in-sample residual scale of the ridge fits plus the kernel-weighted IQR,
averaged the same way and halved. Quantiles are
`q_tau = p1 * exp(median + z_tau * sigma)`.

Guards (same as v1): log hour-1 print capped at the comparables' 90th
percentile (each ensemble uses its own comparables' cap); each median
clamped to the comparables' 5th–95th percentile of ratios at that hour;
per-hour baseline fallback if < 8 comparables have that hour's ratio;
whole-case baseline fallback on any failure; output validated against
the contract (23 rows, hours 2–24, quantiles finite, >0, ≤1e9,
non-decreasing) before it is written.

## Local practice results
Scored against SullyGnome's hourly counts for the four practice launches
(same harness as v1: `score.py` in the scratch dir).

| case | launch | v2 loss | v1 loss | typical miss | 90% coverage |
|------|--------|---------|---------|--------------|--------------|
| p00 | Marvel's Wolverine | 0.1365 | 0.1346 | 25.2% | 0.913 |
| p01 | Red Dead Redemption 2 | 0.0960 | 0.1135 | 15.5% | 0.957 |
| p02 | God of War Ragnarök | 0.1060 | 0.1406 | 19.9% | 0.957 |
| p03 | Marvel's Spider-Man 2 | 0.2239 | 0.2094 | 37.4% | 0.652 |
| **mean** | | **0.1406** | **0.1495** | | **0.870** |

- House baseline on the same cases: **0.387**.
- Practice board: top 1 = 0.0975, top-20 line = 0.122. v2's 0.1406 is
  still above the top-20 line, but beats v1 by 0.009 (6%).
- The gain comes from p01 (−0.018) and p02 (−0.035); p00 (+0.002) and
  p03 (+0.015) get slightly worse. p03's midday dip (hours 10–14,
  log-errors −0.29…−1.06) remains unpredictable from in-case features.
- Runtime: ~0.17 s per case (limit 8 s). Output 1.5 KB (limit 65,536).
- Determinism: two runs byte-identical, including under `python3 -P -s`
  (the grader's invocation).

## Honest cross-checks (not practice answers)
- **Rolling-subset comparable-LOO** (12 held-out rolling launches, the
  final's regime): v2 **0.231** vs v1 **0.233** — the practice gain is
  not bought by overfitting the practice cases.
- **All-comparables LOO** (37 held-out): v2 0.519 vs v1 0.469 — worse,
  but that yardstick scores mostly non-rolling regimes the final is not
  in, and the LOO harness forces a rolling target on them; the
  rolling-subset LOO is the relevant one for GTA VI.

## What was tried and rejected (practice mean)
- Ridge penalty sweep: 0.1 → 0.151, 1.0 → 0.150, 3.0 → 0.148,
  10.0 → 0.145, 30 → 0.152, 100 → 0.184. (Optimum ~10, but it does
  not combine with the roll boost; kept at 1.0.)
- Winsorised ridge targets: 0.150–0.155 (rejected).
- New features from a screen on comparables: ramp48 (log last-48h mean /
  14d mean, corr −0.35…−0.69), ramp7 (−0.56…−0.71), trend, pre_vol,
  year, logch1. In the ridge they add ~nothing once logh1 + logpre are
  in (r7: 0.1483; r48: 0.1496) — collinear with the size features.
  Comparable-LOO confirms only r7 helps slightly (0.443 vs 0.457).
- Fourier shape model (6 shape params per comparable, ridge-regressed):
  0.376 (rejected — unstable).
- Shape-amplitude model (v1): 0.207 (rejected).
- Empirical shape adjustment (rolling-median minus all-median curve,
  smoothed or Fourier): 0.149–0.179 (rejected — the rolling median
  curve is higher at midday and lower in the evening, the opposite of
  what practice needs; the big-rolling median is far worse, 0.17–0.40).
- Rolling LOO-bias adjustment on top of v1 (×1.0: 0.184, ×0.5: 0.153):
  rejected — double-counts the ridge's rolling correction.
- Rolling-only ensemble: 0.172 (rejected — 12 points is too thin).
- Asymmetric spread (separate up/down residual scales): 0.147 (small
  gain on v1, not carried into v2 for simplicity).
- Quantile regression (direct pinball): not built — with 38 points the
  0.05/0.95 tail fits would be noise; the L1 median regression alone
  scored 0.231 in v1's search.

## The in-sample shape correction (NOT in v2 — parent's call)
A per-hour bias correction estimated from the four practice launches'
own answers (mean log-error per hour, applied to the v1 median) would
cut the practice mean to **0.120** (0.124 smoothed) — inside the paid
zone. It is deliberately NOT in v2: it fits on the validation set, so
its practice score would be in-sample and meaningless as a gate, and
its generalisation to GTA VI (a different game, far bigger) is
uncertain. The RULES' design (comparables leave out the practice
launches) treats practice as held-out validation. If the parent wants
the extra ~0.02 and accepts that risk, it is a one-line change
(add the smoothed bias vector to the median); say so explicitly and
v2 can be rebuilt with it. v2 as built uses only the case's own
comparables.

## Robustness tests (all passed)
- Empty comparables → baseline fallback, valid output.
- Hour-1 print = 1 → valid output.
- Missing `known` entirely → baseline with p1 = 1, valid output.
- Mega hour-1 print 850,000 → valid output; peak q50 973,509 at
  hour 17 (≈1.15× hour 1), no explosion past the cap.
- Hour 23 stripped from all but 3 comparables → hour 23 exactly the
  baseline row, other 22 hours keep the ensemble.
- `streams` / `categories_hourly` present (non-null) → ignored, valid.
- `python3 -P -s main.py` (grader invocation): exit 0, output
  byte-identical to a normal run.

## Files
- `main.py` — the entry (single file, pure stdlib).
- `BUILD-REPORT.md` — this file.
- Local scoring harness (not part of the entry, kept outside the repo):
  `~/workspace/makefun/gta6-twitch-ccu-forecast-01-scratch/` — `score.py`,
  `models7.py` (v2 candidates V2–V10), `models8.py` (feature screen +
  ramp features + comparable-LOO evaluators).

## Kit check (local, `makefun check`, 2026-11-17 ~18:35 WIB)
Commit `fcab142361a0680304dbed080c91e13e4ac8e80e`: **11 passed, 2 failed**,
5 skipped, 5 at-submission checks (GitHub repo, parent's step).
Passed: competition_open, snapshot, license_osi, entrypoint,
transcripts_present, transcripts_recorded, thinking_wellformed,
thinking_in_window, thinking_reproduces (36.6% authored, main.py 62
authored lines), thinking_turns, thinking_revised (main.py revised in
3 messages — the refactor below counts as the revision).
Failed, both parent-side: `thinking_priced` (0 usage rows — a
subagent's usage row is written only once its run completes, so the
parent pulls it from muse.db after this run and re-bundles with
`--muse-usage`) and `transcripts_run_record` (needs the bundle's run
record, written by `makefun bundle`).

## Revision note (commit fcab14)
`in_sample_sigma` now takes the ridge fit the caller already computed
for its prediction, so each hour fits each ridge once instead of twice.
Same math, byte-identical outputs, practice mean unchanged at 0.140604.

## Decision for the parent
Practice mean **0.1406** vs v1's **0.1495** and the top-20 line **0.122**:
v2 clearly beats v1 on practice (−0.009, −6%) and is neutral-to-better
on rolling-subset comparable-LOO, so it is a real improvement, not
practice overfitting. It is NOT below the 0.122 paid-zone line on
practice. Whether to replace v1 (Force_Last — submitting v2 replaces
v1's entry) is the parent's call; the entry is built, tested, and ready
to bundle. The only known route below 0.122 on practice is the
in-sample shape correction above, which needs an explicit parent
decision.
