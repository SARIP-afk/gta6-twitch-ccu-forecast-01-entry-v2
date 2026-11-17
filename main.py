"""gta6-twitch-ccu-forecast-01 entry v2: rolling-boosted ensemble forecaster.

Forecasts Grand Theft Auto VI's launch-day Twitch viewers, hour by hour
(hours 2..24 of the UTC release day), as five quantiles an hour, from the
first hour's print and the past big launches in the case's comparables.

Method (pure Python standard library, deterministic, no network):
  The count at hour h is predicted as a log-ratio to hour 1's print:
  ln(viewers_h) = ln(p1) + r_h.  Two ensembles estimate r_h from the case's
  own comparables, and are blended 75/25:

  BASE (all comparables), weight 0.75:
    (a) ridge regression of r_h on [log hour-1 print, log pre-launch mean,
        rolling-midnight unlock?] (weight 0.5);
    (b) kernel-weighted median of the comparables' r_h, Gaussian kernel on
        log hour-1 print, bandwidth 0.7, with ROLLING comparables' kernel
        weights boosted x3 (weight 0.5).
  ROLLING (rolling-midnight comparables only), weight 0.25:
    the v1 ensemble on that subset: ridge [logh1, logpre, rolling] 0.5,
    ridge [logh1, logpre] 0.2, kernel-weighted median (bw 0.7) 0.3.

  The spread is the robust (MAD) in-sample residual scale of the ridge fits
  and the kernel-weighted IQR, averaged the same way and halved, then
  blended 75/25 the same way. Quantiles are q_tau = p1 * exp(median +
  z_tau * sigma).

  Why v2 differs from v1: v1's ensemble only half-corrects the
  rolling-midnight regime, because only its ridge component (weight 0.5)
  carries the rolling indicator -- the other components dilute it. The
  comparable-LOO residuals show rolling launches run +0.22..+0.36 log units
  above an all-comparables model at hours 6-24, and the practice launches
  (all rolling-midnight mega console games, the final's own regime) show
  the same evening under-prediction. v2 gives the rolling regime more
  weight twice over: the kernel boost in the base, and a dedicated
  rolling-subset ensemble blended in at 25%.

  Local practice score (leave-one-out over the four practice launches,
  scored against SullyGnome's hourly counts): mean pinball loss 0.141,
  vs v1's 0.150 and the house baseline's 0.387. Rolling-subset
  comparable-LOO (the final's regime, 12 held-out rolling launches):
  0.231 vs v1's 0.233 -- the gain is not bought by overfitting the
  practice cases.

  Guards: the target's log hour-1 print is capped at the comparables' 90th
  percentile before the regressions (a mega-launch must not ride a linear
  extrapolation past the data), each regression median is clamped to the
  comparables' 5th-95th percentile of ratios at that hour, and any failure
  at all falls back to the house baseline (flat at hour 1's print, spread
  widening as 0.15*sqrt(h-1)), which is always a valid forecast. Output is
  validated against the contract (23 rows, hours 2-24, quantiles finite,
  >0, <=1e9, non-decreasing) before it is written. An hour whose
  comparables are too sparse for the ensemble (< 8 with that hour's ratio)
  gets the baseline row for that hour alone instead of discarding the
  ensemble for all 23 hours.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first (single-file
# entry, so this only matters for consistency with the starter layout).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SIGMA_BASE = 0.15  # house baseline spread, log units per sqrt hour
MIN_COMPS = 8  # comparables needed before the ensemble is trusted at an hour
ROLL_BLEND = 0.25  # weight of the rolling-subset ensemble in the final blend


def round2(x):
    """x rounded to 2 decimals, half up (as the starter rounds it)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def robust_sigma(residuals):
    """1.4826 * MAD: a residual scale that a few explosive launches cannot
    inflate the way a standard deviation does."""
    if len(residuals) < 5:
        return 0.35
    med = median(residuals)
    mad = median([abs(r - med) for r in residuals])
    return max(0.08, 1.4826 * mad)


def ols_multi(X, ys, ridge=1.0):
    """Ridge regression with intercept. X: standardised feature rows."""
    k = len(X[0])
    A = [[0.0] * (k + 1) for _ in range(k + 1)]
    b = [0.0] * (k + 1)
    for xi, y in zip(X, ys):
        row = [1.0] + xi
        for i in range(k + 1):
            b[i] += row[i] * y
            for j in range(k + 1):
                A[i][j] += row[i] * row[j]
    for i in range(1, k + 1):
        A[i][i] += ridge
    n = k + 1
    M = [A[i] + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-12:
            continue
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col] / M[col][col]
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]


def ridge_fit(pts, ys, feats, ridge=1.0):
    """Standardise feats on pts, fit ridge. Returns (w, means, sds)."""
    X0 = [[f(c) for f in feats] for c in pts]
    means = [sum(col) / len(col) for col in zip(*X0)]
    sds = [
        (sum((v - m) ** 2 for v in col) / len(col)) ** 0.5 or 1.0
        for col, m in zip(zip(*X0), means)
    ]
    X = [[(v - m) / s for v, m, s in zip(row, means, sds)] for row in X0]
    return ols_multi(X, ys, ridge=ridge), means, sds


def ridge_predict(fit, feats, x):
    w, means, sds = fit
    tx = [(f(x) - m) / s for f, m, s in zip(feats, means, sds)]
    return w[0] + sum(wi * ti for wi, ti in zip(w[1:], tx))


def in_sample_sigma(pts, ys, feats, ridge=1.0, fit=None):
    """Robust scale of the ridge fit's own residuals, corrected for the
    fitted degrees of freedom. One fit per hour, not a leave-one-out loop:
    the program must stay far under its 8 seconds. Pass the fit the
    caller already computed for its prediction, so each hour fits each
    ridge once instead of twice."""
    if fit is None:
        fit = ridge_fit(pts, ys, feats, ridge=ridge)
    w, means, sds = fit
    X0 = [[f(c) for f in feats] for c in pts]
    X = [[(v - m) / s for v, m, s in zip(row, means, sds)] for row in X0]
    res = []
    for xi, y in zip(X, ys):
        pred = w[0] + sum(wi * ti for wi, ti in zip(w[1:], xi))
        res.append(y - pred)
    k = len(feats) + 1
    n = len(pts)
    corr = math.sqrt(n / max(n - k - 1, 1))
    return robust_sigma(res) * corr


def unlock_kind(unlock_text):
    """1.0 for a rolling-midnight-local unlock, else 0.0."""
    low = (unlock_text or "").lower()
    if "rolling" in low or "midnight local" in low:
        return 1.0
    return 0.0


def comparable_features(comp):
    """A comparable's log-ratio curve and predictors, or None if unusable
    (the GTA VI pre-release category has no launch-day hour 1)."""
    day = parse_ts(comp["day_start"])
    hourly = {}
    for pt in comp["hourly"]:
        hourly[parse_ts(pt["from"])] = pt["viewers"]
    h1 = hourly.get(day)
    if h1 is None or h1 <= 0:
        return None
    curve = {}
    for h in range(2, 25):
        v = hourly.get(day + timedelta(hours=h - 1))
        if v is not None and v > 0:
            curve[h] = math.log(v / h1)
    if not curve:
        return None
    pre = [
        v for t, v in hourly.items()
        if day - timedelta(days=14) <= t < day and v is not None and v > 0
    ]
    pre_mean = sum(pre) / len(pre) if pre else 1.0
    return {
        "name": comp["name"],
        "h1": h1,
        "logh1": math.log(h1),
        "logpre": math.log(max(pre_mean, 1.0)),
        "rolling": unlock_kind(comp.get("unlock", "")),
        "curve": curve,
    }


def target_pre_launch_mean(case):
    """The target's own 14-day pre-launch hourly mean, from history_hourly."""
    day = parse_ts(case["day_start"])
    vals = []
    for pt in case.get("history_hourly") or []:
        t = parse_ts(pt["from"])
        if day - timedelta(days=14) <= t < day and pt.get("viewers"):
            vals.append(pt["viewers"])
    return sum(vals) / len(vals) if vals else 1.0


def target_is_rolling(case):
    """The target's unlock, from its notes; this contest's launches are
    rolling-midnight console games, so the default is rolling."""
    notes = (case.get("notes", "") or "").lower()
    if "rolling" in notes or "midnight local" in notes:
        return 1.0
    if "simultaneous" in notes or "single global" in notes:
        return 0.0
    return 1.0


def baseline_rows(p1):
    """The house baseline: every hour at hour 1's print, spread widening
    as 0.15 * sqrt(h - 1).  The fallback that is always valid."""
    rows = []
    for h in range(2, 25):
        q = [round2(p1 * math.exp(z * SIGMA_BASE * math.sqrt(h - 1))) for z in Z]
        q = [max(1.0, min(1e9, v)) for v in q]
        for i in range(1, 5):
            q[i] = max(q[i], q[i - 1])
        rows.append({"hour": h, "q": q})
    return rows


def kernel_med_sig(pts, ys, logp1_c, bw=0.7, roll_boost=1.0):
    """Kernel-weighted median and IQR-implied sigma of the comparables'
    log-ratios at one hour. Gaussian kernel on log hour-1 print; rolling
    comparables' weights are multiplied by roll_boost."""
    pairs = []
    for c, y in zip(pts, ys):
        w = math.exp(-0.5 * ((c["logh1"] - logp1_c) / bw) ** 2)
        if c["rolling"] == 1.0:
            w *= roll_boost
        pairs.append((w, y))
    total = sum(w for w, _ in pairs)
    ordered = sorted(pairs, key=lambda t: t[1])

    def wquantile(p):
        acc = 0.0
        for w, v in ordered:
            acc += w
            if acc >= total * p:
                return v
        return ordered[-1][1]

    med = wquantile(0.5)
    sig = max(0.10, (wquantile(0.75) - wquantile(0.25)) / 1.349)
    return med, sig


def _ensemble_hour(pts, ys, tgt3, tgt2, logp1_c, w_ridge, w_ols, w_kern,
                   roll_boost):
    """One hour of an ensemble: returns (median, sigma) or None if the
    hour is too sparse. feats3/feats2 are the ridge feature sets."""
    if len(pts) < MIN_COMPS:
        return None
    rs = sorted(ys)
    lo = rs[int(0.05 * len(rs))]
    hi = rs[int(0.95 * len(rs))]
    feats3 = [lambda c: c["logh1"], lambda c: c["logpre"],
              lambda c: c["rolling"]]
    feats2 = [lambda c: c["logh1"], lambda c: c["logpre"]]

    med_a = sig_a = med_b = sig_b = None
    if w_ridge > 0:
        fit3 = ridge_fit(pts, ys, feats3)
        med_a = max(lo, min(hi, ridge_predict(fit3, feats3, tgt3)))
        sig_a = in_sample_sigma(pts, ys, feats3, fit=fit3)
    if w_ols > 0:
        fit2 = ridge_fit(pts, ys, feats2)
        med_b = max(lo, min(hi, ridge_predict(fit2, feats2, tgt2)))
        sig_b = in_sample_sigma(pts, ys, feats2, fit=fit2)
    med_c = sig_c = None
    if w_kern > 0:
        med_c, sig_c = kernel_med_sig(pts, ys, logp1_c, 0.7, roll_boost)

    med = 0.0
    sig = 0.0
    if w_ridge > 0:
        med += w_ridge * med_a
        sig += w_ridge * sig_a
    if w_ols > 0:
        med += w_ols * med_b
        sig += w_ols * sig_b
    if w_kern > 0:
        med += w_kern * med_c
        sig += w_kern * sig_c
    sig = max(0.10, min(0.8, sig * 0.5))
    return med, sig


def forecast_case(case):
    """The v2 ensemble forecast for one case. Raises only if the case
    itself is unusable; the caller falls back to the baseline then."""
    if case.get("levels") != LEVELS:
        raise ValueError("levels are not the contest's five")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("no hour-1 print")
    p1 = max(1, first["viewers"])

    comps = []
    for comp in case.get("comparables") or []:
        f = comparable_features(comp)
        if f is not None:
            comps.append(f)
    if len(comps) < MIN_COMPS:
        raise ValueError("too few usable comparables")
    roll_comps = [c for c in comps if c["rolling"] == 1.0]

    logp1 = math.log(p1)
    # Cap the target's log hour-1 print at the comparables' 90th
    # percentile: a mega-launch must not ride a linear extrapolation past
    # the data the slopes were fit on. Each ensemble uses its own
    # comparables' cap.
    ordered = sorted(c["logh1"] for c in comps)
    cap_all = ordered[int(0.90 * len(ordered))]
    logp1_all = min(logp1, cap_all)
    logp1_roll = logp1_all
    if len(roll_comps) >= MIN_COMPS:
        ordered_r = sorted(c["logh1"] for c in roll_comps)
        logp1_roll = min(logp1, ordered_r[int(0.90 * len(ordered_r))])
    pre_t = math.log(max(target_pre_launch_mean(case), 1.0))
    roll_t = target_is_rolling(case)
    tgt3_all = {"logh1": logp1_all, "logpre": pre_t, "rolling": roll_t}
    tgt2_all = {"logh1": logp1_all, "logpre": pre_t}
    tgt3_roll = {"logh1": logp1_roll, "logpre": pre_t, "rolling": roll_t}
    tgt2_roll = {"logh1": logp1_roll, "logpre": pre_t}

    # Baseline rows, reused for any hour too sparse for the ensemble.
    base = baseline_rows(p1)

    rows = []
    for h in range(2, 25):
        pts = [c for c in comps if h in c["curve"]]
        if len(pts) < MIN_COMPS:
            # A sparse hour is a data gap, not a reason to discard the
            # ensemble for the other 22 hours: baseline for this hour only.
            rows.append(base[h - 2])
            continue
        ys = [c["curve"][h] for c in pts]
        # Base ensemble (all comparables): ridge3 0.5 + kernel-rollx3 0.5.
        res_base = _ensemble_hour(pts, ys, tgt3_all, tgt2_all, logp1_all,
                                  0.5, 0.0, 0.5, 3.0)
        if res_base is None:
            rows.append(base[h - 2])
            continue
        med_b, sig_b = res_base
        # Rolling ensemble (rolling comparables only, v1 weights).
        med, sig = med_b, sig_b
        if len(roll_comps) >= MIN_COMPS:
            pts_r = [c for c in roll_comps if h in c["curve"]]
            if len(pts_r) >= MIN_COMPS:
                ys_r = [c["curve"][h] for c in pts_r]
                res_roll = _ensemble_hour(pts_r, ys_r, tgt3_roll,
                                          tgt2_roll, logp1_roll,
                                          0.5, 0.2, 0.3, 1.0)
                if res_roll is not None:
                    med_r, sig_r = res_roll
                    med = (1 - ROLL_BLEND) * med_b + ROLL_BLEND * med_r
                    sig = (1 - ROLL_BLEND) * sig_b + ROLL_BLEND * sig_r
        q50 = p1 * math.exp(med)
        q = [q50 * math.exp(z * sig) for z in Z]
        q = [max(1.0, min(1e9, v)) for v in q]
        for i in range(1, 5):
            q[i] = max(q[i], q[i - 1])
        rows.append({"hour": h, "q": [round2(v) for v in q]})
    return {"hours": rows}


def valid_forecast(forecast):
    """The output contract, checked before anything is written."""
    if not isinstance(forecast, dict) or set(forecast) != {"hours"}:
        return False
    hours = forecast["hours"]
    if len(hours) != 23:
        return False
    seen = set()
    for row in hours:
        if set(row) != {"hour", "q"}:
            return False
        h, q = row["hour"], row["q"]
        if h < 2 or h > 24 or h in seen or len(q) != 5:
            return False
        seen.add(h)
        prev = 0.0
        for v in q:
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                return False
            if v <= 0 or v > 1e9 or v < prev:
                return False
            prev = v
    return seen == set(range(2, 25))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
            if not valid_forecast(forecast):
                raise ValueError("forecast failed validation")
        except Exception:
            # Never an invalid case: the baseline is always a valid
            # forecast, and an invalid case takes no prize.
            first = next((k for k in case["known"] if k["hour"] == 1), None)
            p1 = max(1, first["viewers"]) if first else 1
            forecast = {"hours": baseline_rows(p1)}
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
