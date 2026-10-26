# Stage 1 Target-Domain Report — NLR ESIF Facility Power

> Task book: *PowerForecast 算力中心中短期负荷预测模型开发任务书 V2.0*, Stage 1 (§3), time
> numbering (§5.1–5.2), tests (§9.2/§9.5), Milestone 2 (§11), deliverables (§12/§13).
> Branch: `research/power-transfer-v1` (no commit/push/reset performed).
> Implementation: `src/data/target_power.py`, `scripts/prepare_target_power.py`,
> `scripts/analyze_power.py`, `configs/datasets/nlr_target.yaml`,
> `tests/test_target_power.py`, `tests/test_preprocessing.py` (this report).

## 1. Scope and honesty statement

This report covers **only the target domain** (real NLR ESIF facility power) of Stage 1.
External source-domain data, feature alignment and the Stage 2 power main model are other
workstreams and are not claimed here.

The target label is

```text
facility_power_kw = it_power_kw + cooling_kw + hvac_kw + plug_and_light_kw + pump_kw
```

a **component-sum derived facility power** label. Its declared label type is
**`facility_derived_kw`** (not `facility_measured_kw`): the raw archive contains no
independent facility total-meter channel. The five components are the label composition
and must never be used as future-known features when predicting the label they
reconstruct. This is recorded in every artefact (`target_metadata.json`), enforced by
`src.data.labels.classify_target` / `require_main_model_label`, and covered by tests.

No synthetic data is used anywhere in this workstream: every number and figure below is
computed from the two read-only raw NLR parquet files.

## 2. Inputs (read-only)

| role | file | rows | size | SHA256 (streamed) |
|---|---|---|---:|---|
| power label source | `data/raw/NLR_Dataset/esif.influx.buildingData.PUE.combined.parquet` | 4,569,543 | 109,685,321 B | stored in `target_metadata.json` |
| target-domain weather | `data/raw/NLR_Dataset/esif.influx.buildingData.outside.combined.parquet` | 4,034,917 | 58,865,468 B | stored in `target_metadata.json` |

Raw files are opened read-only. Their size and modification time are identical before and
after the pipeline; the test suite asserts this explicitly. Nothing is written below
`data/raw/` (writes are refused before any file is created).

## 3. Time conventions (explicit, provisional)

| item | value |
|---|---|
| declared timezone | `UTC` |
| `timezone_verified` | **`false`** — the site local timezone is NOT verified |
| raw timestamp handling | raw influx timestamps are timezone-naive; they are stored as naive UTC (project convention) |
| trace grid | anchor-based 96-slot / 24-hour trace days, `t0 + k×96` 15-minute slots |
| local-time / DST operations | none; `DatasetSpec.require_verified_timezone()` blocks any downstream operation that would need them |
| interval semantics | label = interval **start** `t`, interval `[t, t+15 min)` |

Because every day/slot computation is anchor-based, Stage 1 never needs the unverified
local timezone. This is a deliberate provisional assumption, not an unstated UTC claim.

## 4. Anchor and data selection (no gap compression)

* Raw series range: 2015-11-10 03:00:01 → 2025-08-29 04:35:08 (naive UTC).
* The five power components are **all NaN before 2016-06-12 06:00 UTC**:
  136,484 raw rows, 0% finite component values.
* First raw sample with all five components finite: **2016-06-12 06:00:18 UTC**; its
  15-minute interval starts at **2016-06-12 06:00:00 UTC**.
* Prepared trace-grid anchor `t0 = 2016-06-12 06:00:00 UTC` (declared in
  `configs/datasets/nlr_target.yaml: analysis.window_start`). The 136,484 pre-anchor rows
  are excluded with explicit counts in metadata/quality (documented selection boundary,
  because they cannot produce a label); they are **not** silently dropped.
* Grid: **323,035 slots** from 2016-06-12 06:00:00 to 2025-08-29 04:30:00
  (3,365 trace days; the last trace day has 91 of 96 slots).
* Missing intervals inside the range keep their grid rows (`power_kw = NaN`,
  `power_valid = false`); later timestamps never shift. The axis is never compressed.
* `analysis.window_end` is supported as an exclusive bound but is `null` for the canonical
  full run.

## 5. 15-minute aggregation and validity definition

Native sampling is ~60 s (5–95% spacing: 58–65 s, median 60 s). Declared expectation:
**15 native samples per 15-minute interval** (audited, not assumed).

```text
coverage_ratio = min(#native samples whose five components are ALL finite, 15) / 15      ∈ [0, 1]
component interval mean = arithmetic mean of the finite native samples in [t, t+15 min)
facility_power_kw       = sum of the five component interval means, min_count=5
power_valid             = all five component interval means finite
                          AND coverage_ratio ≥ coverage_validity_threshold
```

Every component/feature missing value becomes NaN (never zero), infinite values are
replaced by NaN and counted, duplicate raw timestamps are collapsed deterministically by the
component-wise mean, and no forward/backward fill or imputation is ever applied to the
label.

**Coverage threshold choice — `1/15 = 0.0667` (declared in YAML).** The raw archive
normally provides 15 samples per interval, but short logger windows sample every ~5 minutes:
inside the verified first 31 trace days, 242 of 2,976 intervals contain only 2–14 samples.
A 0.8 threshold would mark those real observations invalid (2,734/2,976 valid), contradicting
the verified Stage-1 acceptance fact that the first 31 trace days are 100% covered. The
threshold therefore encodes **"at least one expected native sample present"** — a
*presence* gate, not a quality gate. The actual per-slot `coverage_ratio` is stored, and
downstream training/evaluation must use it for quality-aware weighting. This limitation is
reported openly.

## 6. Data quality results (full prepared range)

| metric | value |
|---|---:|
| grid slots | 323,035 |
| valid labels (`power_valid=true`) | **310,367 (96.08%)** |
| invalid labels | 12,668 |
| intervals with no native sample at all | 12,663 |
| intervals with native samples but no single sample having all five components finite | 5 (invalid) |
| `coverage_ratio` mean / median | 0.9144 / 1.0 |
| intervals with full 15-sample coverage | 86.93% |
| intervals with `coverage_ratio ≥ 0.8` | 89.47% |
| intervals with `coverage_ratio = 0` | 3.92% |
| native samples/interval: mean / median / p05 / p95 | 13.72 / 15 / 3 / 15 |
| intervals with the expected 15 samples | 86.93% |
| raw component samples replaced (infinite) | 0 |
| duplicate raw timestamps | 0 |
| excluded pre-anchor raw rows (all NaN) | 136,484 |
| negative component raw samples (kept, not clipped) | cooling 113, hvac 428, pump 475 |
| interval means outside declared physical ranges | 0 |
| facility power area out of declared range | 0 |

Valid facility power over the full range: mean **1,540.41 kW**, std 817.94, median 1,224.63,
min 14.21, max 4,171.17 kW. The long-term level shifts (≈0.8 MW in 2016, ≈1.5 MW in 2019,
≈3.5 MW in 2024, ≈2.7 MW in 2025) are a real property of this facility, not a processing
artefact; they dominate the raw-level ACF (see §8.4).

### 6.1 Verified first 31 trace days (acceptance test)

From the first valid slot `2016-06-12 06:00:00` to `2016-07-13 05:45:00`
(96 points/day, 31×96 = 2,976 slots):

| metric | value |
|---|---:|
| prepared slots | **2,976 / 2,976** |
| valid labels | **2,976 / 2,976 (100%)** |
| facility power mean | **820.18 kW** |
| facility power min / max | 606.55 / 950.63 kW |
| intervals with full 15-sample coverage | 2,724 / 2,976 (the rest are 2–14-sample real observations, still valid) |

These numbers are asserted by the real-raw integration test, so a regression in the
selection/anchor/aggregation path fails CI.

## 7. Artefacts

All under `data/processed/` (regenerated atomically by the prepare script; declared names):

| file | contents / role |
|---|---|
| `processed_target_power.parquet` | 323,035 × **5 columns** `timestamp, site_id, power_kw, power_valid, coverage_ratio`; one row per 15-min interval start; naive UTC |
| `target_components.parquet` | 323,035 × 6 `timestamp + five component interval means (kW)`; **explicitly flagged label-composition-only** in metadata (must not be future-known features) |
| `processed_target_features.parquet` | **LONG** format 646,070 rows × 7 `timestamp, domain_id, feature_name, feature_value, unit, observed_mask, coverage_ratio`; features `outdoor_air_temp_c` (°C) and `relative_humidity_pct` (%); role `target_only`, availability `historical_observed`, `future_mapping.local_weight = 1.0` recorded in metadata but **not used in Stage 1** |
| `target_metadata.json` | label type/origin/boundary/warning, target spec, timezone assumption + flag, anchor, selection counts, aggregation rules, quality summary, field registry, source file hashes, provenance |
| `target_quality.json` | full quality audit (grid, native sampling, coverage, validity, components, physical-range review, selection, first-31-day window) plus artefact SHA256s |

Target-domain weather features (metadata only, no model use): Fahrenheit→Celsius conversion
`T[°C] = F × 0.5555555555555556 − 17.77777777777778` (documented historical config; verified
against the legacy processed file to <1e-9). Raw `unit` column audit: 3,306,781 null /
728,136 `"outside"` — it is recorded but explicitly **not trusted** as a unit declaration.
The 1970-01-01 sentinel row is excluded (1 row, recorded). Weather observed slots:
temperature 299,641, humidity 299,643 of 323,035; mean coverage 0.818; converted temperature
range −39.67…40.13 °C; humidity 0–100%; 0 out-of-declared-range values.

## 8. Power characteristics analysis

Output directory: `results/analysis/stage1/power_characteristics/` — all figures are
English-titled, carry the "REAL — NLR ESIF facility_derived_kw" label, and use the
matplotlib Agg backend. Numeric summaries live in `power_stats.json`.

### 8.1 Figures

| file | content |
|---|---|
| `raw_power_curve.png` | full prepared series, decimated ×16 for rendering only |
| `typical_day_profile.png` | mean/median by 15-min slot, weekday vs weekend, plus deviation panel |
| `hourly_daily_weekly.png` | hourly mean, day-of-week mean, weekly mean |
| `autocorrelation.png` | level ACF and 15-min-increment ACF, lags 0…672 (7 days) |
| `ramp_volatility.png` | 15-min ramp ΔkW histogram + 24-h rolling standard deviation |
| `peak_valley.png` | per-trace-day peak / valley / range time series + range histogram |
| `coverage_quality.png` | daily valid fraction and daily mean `coverage_ratio` |

### 8.2 Level, peak/valley and ramp statistics

| metric | value |
|---|---:|
| facility power mean / std | 1,540.41 / 817.94 kW |
| daily (trace-day) peak mean | 1,644.78 kW |
| daily valley mean | 1,414.15 kW |
| daily peak–valley range mean / median / p95 / max | 230.63 / 158.04 / 688.11 / 3,491.51 kW |
| eligible days for daily statistics (≥75% valid slots) | 3,195 / 3,365 |
| 15-min ramp ΔkW: mean / std | 0.04 / 31.57 kW |
| 15-min ramp |ΔkW|: mean / p95 / p99 | 9.57 / 20.17 / 98.05 kW |
| ramp extremes: max positive / max negative | +2,219.84 / −2,240.95 kW |
| 24-h rolling std: mean / median / p95 / max | 62.16 / 39.05 / 200.36 / 1,673.30 kW |
| weekly block mean: mean / min / max | 1,539.06 / 344.51 / 3,810.25 kW |

### 8.3 Daily / weekly / seasonal profiles

* Hourly mean range: 1,518.47–1,556.74 kW (peak-to-peak 38.28 kW).
* Day-of-week mean (trace-day weekday): Mon 1,496 → Thu 1,574 → Sun 1,512 kW.
* Monthly mean: Apr 1,624, May 1,660, Jul 1,510, Aug 1,452 kW (seasonal swing ≈ 208 kW).
* Weekday slot-profile peak-to-peak: 60.39 kW = **7.4% of the level standard deviation**.
* Weekday and weekend 15-min profiles differ smoothly (weekend starts higher in the early
  morning, weekday higher in the evening); the per-timestamp weekday artifact at UTC
  midnight was removed by classifying weekday/month by **trace-day start**.

### 8.4 Autocorrelation — honest interpretation

The level ACF decays very slowly: 0.9993 (1 step), 0.9650 (1 day), 0.9002 (7 days). This is
**not** a diurnal periodicity; it reflects persistent multi-year level shifts. The
first-difference (15-min increment) ACF is essentially zero at every lag (lag 1 = 0.015,
lag 96 = −0.006, |ACF| < 0.08 everywhere), and the mean daily profile amplitude is small
relative to the level spread. No separate 24-hour or 7-day periodicity peak stands above
the local trend. This is a real property of this facility's load; it means Stage 2 must
still evaluate the naive seasonal baselines (B0/B1) explicitly rather than assume a strong
diurnal cycle carries the model.

## 9. Reproduce

```powershell
# tests for this workstream (fast)
& .\.venv\Scripts\python.exe -m pytest tests/test_target_power.py tests/test_preprocessing.py -q

# full Stage 1 target build -> data/processed/ (reads raw, writes 5 artefacts)
& .\.venv\Scripts\python.exe -m scripts.prepare_target_power --config configs/datasets/nlr_target.yaml

# power characteristics -> results/analysis/stage1/power_characteristics/
& .\.venv\Scripts\python.exe -m scripts.analyze_power --config configs/datasets/nlr_target.yaml

# bounded debug run (no full-archive read; audit counts are relative to the loaded slice)
& .\.venv\Scripts\python.exe -m scripts.prepare_target_power --config configs/datasets/nlr_target.yaml `
    --start 2024-01-01T00:00:00 --end 2024-02-01T00:00:00 --out data/processed_debug
```

The prepare command refuses any `--out` below `data/raw/` and exits non-zero on error.
Re-running the canonical command regenerates the declared artefact names atomically; the
raw archive and unrelated files are untouched.

## 10. Stage 2 hand-off (declared API)

`src/data/target_power.py` exposes:

| function | purpose |
|---|---|
| `load_raw_components(path, *, start=None, end=None, columns=None)` | bounded, streaming pyarrow reader (raw never modified) |
| `build_target_power_frame(raw, config) -> (frame, audit)` | complete anchored 15-min target frame + quality audit |
| `build_target_power_frame_with_components(raw, config) -> (frame, components, audit)` | same + component interval means for the composition-only artefact |
| `build_target_feature_frame(weather_raw, grid, config) -> (long_features, audit)` | weather covariates in the long contract format |
| `target_spec_from_config(config) -> TargetSpec` | `active_power` / `facility` / `kW` / `component_sum` / non-synthetic → label type `facility_derived_kw` |
| `dataset_spec_from_config(config) -> DatasetSpec` | dataset/domain/features/target/anchor with `timezone_verified=false` for `ForecastInputs` wiring |
| `build_target_metadata(frame, quality, config, *, source_metadata=None)` | JSON metadata document |
| `write_target_artifacts(frame, components, features, metadata, quality, config, *, out_dir=None)` | atomic write of the five artefacts, refuses `data/raw/` |

Stage 2 may use `processed_target_power.parquet` (core five columns) directly, construct
`ForecastInputs` from `past_power_kw / past_power_valid` and the target frame timestamps,
and must keep `power_valid` masking (masked labels are NaN, never 0 kW). The main model must
**not** read the weather feature artefact (Stage 1 does not use it; it is registered only
for the future auxiliary path) and must **not** read `target_components.parquet` as
future-known input.

## 11. Tests and verification

* `tests/test_target_power.py` (new, 27 tests): label taxonomy/gate, complete grid without
  gap compression, deterministic duplicate collapse, non-finite → invalid (never zero),
  coverage bounds and threshold monotonicity, exact component-sum equality, weather long
  format + conversion + sentinel, metadata honesty/timezone warning/selection counts,
  writer round-trip and `data/raw/` refusal, raw-file immutability, bounded real-raw loader,
  and the real first-31-day integration check (2,976/2,976 valid, 820.18 kW).
* `tests/test_preprocessing.py` (new, 16 tests): existing `quality_report`,
  `validate_frame` structural rejections, `fill_causal` (forward-only, covariate-only,
  target untouched, bounded limit), `write_processed` versioned output + sidecar,
  `data/raw/` refusal, unlabeled frame handling.
* Both files pass in ≈7 s. The 126 pre-existing skeleton tests still pass: the combined run
  `pytest tests/test_baselines.py tests/test_contracts.py tests/test_leakage.py
  tests/test_lifecycle.py tests/test_protocols.py tests/test_registry.py tests/test_smoke.py
  tests/test_unlabeled.py tests/test_windows.py tests/test_target_power.py
  tests/test_preprocessing.py -q` reports **169 passed in 33 s** (126 skeleton + 43 Stage 1).
* A full `pytest -q` run during development reported 241 passed / 1 failed, where the single
  failure was in `tests/test_model_adapters.py` — a concurrent Stage 2 workstream that was
  being edited at that moment; the same test passes when run alone after that workstream's
  update, and later full-suite runs are dominated by its slow foundation-model tests (minutes
  per test). No Stage 1 target-domain test failed in any run, and no existing file outside
  this workstream's scope was modified.

## 12. Limitations and deviations (reported, not hidden)

1. **Timezone is provisional.** `timezone_verified=false`; no local-time/DST claim is made.
   All trace days are anchor-based. If the site timezone is later verified, the anchor-based
   numbering still holds, but any local-calendar interpretation must be re-checked.
2. **Label is derived, not measured.** `facility_derived_kw` from five sub-meters; no
   independent facility total meter exists in this archive. Results must be reported with
   that label type and never compared as if it were a total-meter measurement.
3. **Coverage threshold is a presence gate (1/15).** It accepts an interval mean built from
   at least one native sample; 5%+ of intervals have sparse sampling. Actual
   `coverage_ratio` is stored per slot and must be used for quality-aware evaluation.
4. **Pre-2016 era excluded by an explicit selection boundary** (136,484 raw rows at 0%
   label coverage), not by deleting internal missing positions.
5. **No strong intraday/weekly periodicity** in the real data; the ACF is dominated by
   level persistence. This is reported honestly instead of inventing a cycle.
6. **Weather `unit` column is unreliable** (mostly null); Fahrenheit is inferred from the
   documented historical conversion and cross-checked against the legacy processed file.
   Weather features are target-only historical observations and are not used in Stage 1.
7. `power_stats.json` reports both level and increment ACF, because the level ACF alone
   would be misleading for this non-stationary series; the task-book requirement of an ACF
   up to 7 days (672 lags) is satisfied for both.
