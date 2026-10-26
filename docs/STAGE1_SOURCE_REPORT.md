# Stage-1 Source-Domain Report — M100 ExaData, UCC 2023 audit, candidate sources

> **Status:** Stage-1 source-domain work implemented and executed on the REAL local data.
> M100 ExaData is processed into the 15-minute long-format source panel; UCC 2023 is
> bounded-audited at its native 10-minute frequency; Alibaba GPU v2020 is registered as a
> candidate and deliberately **not** ingested. **No source-domain dataset is approved for
> training in this round** (see §9).
>
> Code: `src/data/source_features.py`; CLI: `scripts/prepare_source_features.py`,
> `scripts/analyze_sources.py`; config: `configs/datasets/sources.yaml`; tracked registry:
> `configs/datasets/feature_registry.yaml`; tests: `tests/test_source_features.py`.
> Cross-source alignment is reported separately in `docs/FEATURE_ALIGNMENT_REPORT.md`.

## 1. Scope and data roles (task book §3.1)

| Dataset | Role in Stage 1 | Label | Native sampling | Audit | Approved for training? |
|---|---|---|---|---|---|
| **M100 ExaData** (`data/raw/M100/extracted/*.parquet`) | primary REAL source for operational-feature pretraining (`unlabeled_pretrain`, target = `None`) | none used as a target; node power exists as hardware telemetry | 15 min, tz-aware UTC | partial | **no** |
| **UCC 2023** (`data/raw/UCC2023/*.csv`) | secondary REAL source, bounded audit only; **not** panelised | none | 10 min (audited on a bounded subset) | unverified | **no** |
| **Alibaba GPU v2020** (`data/raw/Alibaba-GPU-v2020/*`) | registered candidate, status `not_ingested` | none | not audited | unverified | **no** |
| NLR ESIF (`data/raw/NLR_Dataset/*.parquet`) | **target domain reference only** for field alignment; `target-power` owns the authoritative processed target files | `facility_derived_kw` (component sum) | 15 min | partial | target task, not a source |

M100 is **node-level hardware telemetry**: per-server power rails, CPU/GPU package
temperatures and fan speeds. It is **not facility-level power** and is never reported as a
facility total. The label taxonomy still treats it as an unlabelled pretraining source; the
only place its measured node power is declared is a comparison-only field-alignment spec
(§3 of the alignment report), which is explicitly node-scope and never promoted to facility kW.

## 2. Primary source: M100 ExaData

### 2.1 Files, span and native sampling

* 16 parquet files (`2,3,4,5,6,7,8,9,10,11,13,14,16,17,18,19`), one file = one node/domain;
  total raw size **269,830,131 bytes (269.8 MB)**; 355 columns of hardware telemetry per file.
* Total raw rows: **1,311,025** (per node 72,944 – 83,925).
* Timestamps are **tz-aware UTC** and start at `2020-03-09T12:00Z`. 13 nodes end at
  `2022-09-28T21:45Z`; nodes 13/14/19 end at `2022-09-28T22:00Z` (~2.55 years).
* Native interval audit (`audit_m100_timestamps`): the modal within-file timestamp delta is
  exactly **900 s (15 min)** for every node, every delta is an integer multiple of 900 s,
  there are no duplicate timestamps and the time index is monotonically increasing.
  **Verdict: `15min_native_kept`** — M100 needs no resampling and none is performed.
* Each node has 5–9 telemetry gap events. The largest offline gaps are multi-week
  (≈681 h ≈ 28 d for most nodes; up to ≈1,663 h ≈ 69 d for node 5). Gap events are kept.

### 2.2 Curated field set (12 of 355 columns)

Only the `_avg` statistic of a deliberately curated metric list is registered in Stage 1.
Columns named `<metric>_std/_min/_max` stay raw; `value` and `__index_level_0__` are
bookkeeping columns and are **never** treated as physical features.

| canonical name | raw field | unit | semantic type | interval statistic | aggregation rule | physical range | observed fraction |
|---|---|---|---|---|---|---|---|
| `node_total_power_w` | `total_power_avg` | W | power | interval_mean | native_kept | [0, 3000] | 0.8394 |
| `cpu0_power_w` | `p0_power_avg` | W | power | interval_mean | native_kept | [0, 400] | 0.8394 |
| `cpu1_power_w` | `p1_power_avg` | W | power | interval_mean | native_kept | [0, 400] | 0.8394 |
| `cpu0_memory_power_w` | `p0_mem_power_avg` | W | power | interval_mean | native_kept | [0, 200] | 0.8394 |
| `cpu0_io_power_w` | `p0_io_power_avg` | W | power | interval_mean | native_kept | [0, 200] | 0.8394 |
| `psu0_input_power_w` | `ps0_input_power_avg` | W | power | interval_mean | native_kept | [0, 2000] | 0.8410 |
| `psu1_input_power_w` | `ps1_input_power_avg` | W | power | interval_mean | native_kept | [0, 2000] | 0.8410 |
| `disk_power_w` | `fan_disk_power_avg` | W | power | interval_mean | native_kept | [0, 500] | 0.8394 |
| `node_ambient_temp_c` | `ambient_avg` | °C | temperature | interval_mean | native_kept | [-40, 70] | 0.8410 |
| `gpu0_core_temp_c` | `gpu0_core_temp_avg` | °C | temperature | interval_mean | native_kept | [-20, 120] | 0.8394 |
| `gpu1_core_temp_c` | `gpu1_core_temp_avg` | °C | temperature | interval_mean | native_kept | [-20, 120] | 0.8394 |
| `fan0_0_speed_rpm` | `fan0_0_avg` | RPM | fan_speed | interval_mean | native_kept | [0, 15000] | 0.8410 |

Deferred/refused fields (recorded in config + metadata, not registered as physical features):

* `pcie_avg`, `gpu0_mem_temp_avg` — deferred: `pcie_avg` has no documented unit/physical
  meaning in the local copy; the GPU memory-junction temperature duplicates the curated core
  temperature role. They can be registered later with their own audit.
* `value`, `__index_level_0__` — bookkeeping, excluded.
* Two PSU inputs are **not summed**: a PSU sum would be a derived quantity, not a measured
  field. Likewise the CPU power domains are kept as separate measured channels.

`node_ambient_temp_c` is a **node chassis/inlet** sensor, not site outdoor weather; it is
deliberately given a different canonical name from the target's outdoor `temperature_c`
(see alignment report §2). All physical ranges were set from the observed audit and produced
**0 range violations** on the real data (values are counted, never clipped).

### 2.3 Observed distributions (bounded summary; full stats in `source_quality.json`)

| feature | min | p05 | median | p95 | max |
|---|---|---|---|---|---|
| `node_total_power_w` | 260.0 | 365.3 | 404.4 | 445.8 | 1780.0 |
| `cpu0_power_w` | 4.9 | 21.2 | 30.0 | 53.2 | 170.5 |
| `cpu1_power_w` | 6.0 | 23.3 | 30.8 | 45.1 | 164.1 |
| `cpu0_memory_power_w` | 17.0 | 18.0 | 18.0 | 19.0 | 52.6 |
| `cpu0_io_power_w` | 50.0 | 51.8 | 56.5 | 66.9 | 86.4 |
| `psu0_input_power_w` | 0.0 | 230.9 | 258.9 | 281.1 | 994.3 |
| `psu1_input_power_w` | 0.0 | 260.4 | 290.9 | 328.7 | 1587.9 |
| `disk_power_w` | 27.0 | 30.8 | 32.4 | 34.4 | 254.9 |
| `node_ambient_temp_c` | 11.6 | 18.3 | 23.0 | 26.1 | 45.2 |
| `gpu0_core_temp_c` | 24.0 | 30.1 | 35.2 | 39.0 | 82.0 |
| `gpu1_core_temp_c` | 25.0 | 33.0 | 38.0 | 41.0 | 77.0 |
| `fan0_0_speed_rpm` | 0.0 | 4300.0 | 4342.2 | 4404.4 | 10102.2 |

The node total power is clearly bimodal (≈380 W and ≈460 W clusters) because nodes are not
identical; temperatures are slow-moving, fan speed is nearly constant with rare excursions.

## 3. Long-format source panel contract and cleaning (task book §3.2–3.3)

`build_source_panel` writes exactly the seven logical fields, in this order:

```
timestamp, domain_id, feature_name, feature_value, unit, observed_mask, coverage_ratio
```

There is **no label/target column** (`contains_label_columns=false` is asserted by the code,
by metadata and by the test suite). Cleaning rules actually executed:

1. **Timestamp parsing/sorting**: each file is read as tz-aware UTC, sorted monotonically;
   duplicate timestamps raise `ContractError` (the audited files contain none).
2. **Timezone recorded**: UTC, verified by the raw parquet type; metadata records it.
3. **Per-domain 15-minute grid**: every node is reindexed on **its own** complete 15-minute
   grid from its own first to last native timestamp (89,608 slots, 89,609 for nodes
   13/14/19). Domains are never spliced or paired by date; the audited end-time difference
   between node 10 (`21:45Z`) and node 13 (`22:00Z`) is preserved (tested).
4. **Gaps preserved**: internal telemetry holes become panel rows with
   `observed_mask=false`, `feature_value=NaN`, `coverage_ratio=0.0`. The time axis is never
   compressed by deleting positions.
5. **Missing semantics**: NaN means "missing observation", never a measured zero.
   `observed_mask` is True exactly where `feature_value` is finite; `coverage_ratio` is 1.0
   for an observed native 15-minute sample and 0.0 for a missing one.
6. **Non-finite handling**: `inf/-inf` would be converted to NaN and counted in quality
   metadata; 0 non-finite values were found in the real M100 data.
7. **Physical range audit**: values outside the declared ranges are counted (0 in this run),
   never silently clipped.
8. **Original data immutable**: the pipeline is read-only below `data/raw/`; a test compares
   the whole raw tree before/after a build and the writer refuses raw destinations.

Because the panel is long-format, no cross-domain time pairing exists. The full panel is
**17,204,772 rows** (1,433,731 timestamps × 12 features), 41.5 MB as parquet (compact
categorical/float32 dtypes), ≈344 MB in memory during the build. A per-node bounded build is
available through `--nodes` / `--max-rows-per-node` for quick checks and tests.

## 4. Written artefacts

| artefact | rows / size | purpose |
|---|---|---|
| `data/processed/processed_source_features.parquet` | **17,204,772 rows**, 41,486,588 bytes | the long-format source panel (no labels) |
| `data/processed/source_metadata.json` | 101.3 kB | field semantics/units/ranges/availability, native-frequency audit, missing semantics, file SHA256, UCC/Alibaba audit status, training-approval status |
| `data/processed/source_quality.json` | 33.5 kB | exact coverage/missingness counts, range violations, per-domain timeline audit, deterministic distribution samples |
| `results/analysis/stage1/source_characteristics/` | 6 figures + `source_stats.json` + `alignment.json` | source characteristics and alignment evidence |
| `configs/datasets/feature_registry.yaml` | tracked | the field registry deliverable (§ FEATURE_ALIGNMENT_REPORT) |

Raw-file provenance: streamed SHA256 of all 16 M100 parquet files, both UCC CSVs, and file
sizes for the (not ingested) Alibaba tables are stored in `source_metadata.json`.

## 5. Missingness and coverage

Per-feature observed fractions are 0.8394–0.8410 (mean 0.840) over the full 16-node grid:
missingness has two sources — (a) node offline gaps (5–9 events per node, multi-week at
worst) and (b) NaN samples inside otherwise present rows (~8–9 % of raw rows for most
features). Neither is imputed.

| domain | grid slots | observed slots | mean observed fraction |
|---|---|---|---|
| m100_node_2 | 89,608 | 72,921 | 0.8132 |
| m100_node_3 | 89,608 | 66,403 | 0.7406 |
| m100_node_4 | 89,608 | 75,256 | 0.8394 |
| m100_node_5 | 89,608 | 70,418 | 0.7854 |
| m100_node_6 | 89,608 | 76,298 | 0.8511 |
| m100_node_7 | 89,608 | 77,204 | 0.8612 |
| m100_node_8 | 89,608 | 76,854 | 0.8572 |
| m100_node_9 | 89,608 | 77,142 | 0.8607 |
| m100_node_10 | 89,608 | 77,206 | 0.8610 |
| m100_node_11 | 89,608 | 77,187 | 0.8602 |
| m100_node_13 | 89,609 | 76,855 | 0.8572 |
| m100_node_14 | 89,609 | 77,199 | 0.8609 |
| m100_node_16 | 89,608 | 75,857 | 0.8460 |
| m100_node_17 | 89,608 | 76,858 | 0.8572 |
| m100_node_18 | 89,608 | 76,860 | 0.8530 |
| m100_node_19 | 89,609 | 75,248 | 0.8351 |

Node 3 is the worst observed node (0.741) due to a multi-week offline gap; this is reported,
not fixed by deletion, interpolation or a different time anchor.

## 6. Source feature/characteristics analysis

`scripts/analyze_sources.py` writes (all titles English, `Agg` backend, computed only from
the real processed panel/raw timestamps; gaps stay NaN):

* `m100_feature_distributions.png` — observed-value histogram grid per curated metric
  (bimodal node power; near-constant fan speed; skewed CPU/IO/PSU power).
* `m100_availability_heatmap.png` — observed fraction per feature × node/domain
  (0.74–0.86), i.e. where telemetry actually exists.
* `m100_power_and_temp_timeseries.png` — example domain `m100_node_10`,
  `2021-11-10`–`2021-11-24` (chosen inside its longest gap-free run): node total power vs
  ambient/GPU temperatures on twin axes. A real one-sample anomaly around
  `2021-11-23T18:00Z` spikes all channels simultaneously; it is genuine data and is left as-is.
* `m100_periodicity.png` — slot-of-day z-scored profiles and pairwise-complete
  autocorrelation for `node_total_power_w`, `gpu0_core_temp_c`, `node_ambient_temp_c`
  (temperatures show much stronger short-lag persistence than node power; node power has
  only weak diurnal structure at this aggregation).
* `m100_15min_feasibility.png` — native delta histogram (one 900 s bar dominating) and raw
  samples per 15-minute slot (mostly 1; a substantial count of 0 slots = preserved gaps).
* `ucc_audit.png` — UCC bounded audit (§7).
* `source_stats.json` — rows/domains/features, observed fractions, distribution summary,
  native-frequency and aggregation verdicts, field-alignment counts.
* `alignment.json` — full alignment routes and reasons.

## 7. UCC 2023 bounded audit — native 10-minute frequency preserved

| property | value |
|---|---|
| primary file | `data/raw/UCC2023/dataset1_combined.csv`, 613,212,066 B, **2,748,525 rows**, 18 columns |
| secondary file | `data/raw/UCC2023/dataset2_combined.csv`, 184,178,675 B, **1,281,024 rows**, 13 columns |
| bounded rows audited | first **300,000** rows per file (documented limitation) |
| dataset1 series in subset | 16 (`qh2-rcc120 … qh2-rcc135`), subset span `2018-10-09T11:20 … 2019-03-06T11:00` |
| dataset1 native deltas in subset | 299,934 × 600 s + gap deltas; all multiples of 60 s; no duplicates/negative deltas in the subset |
| dataset1 missing values in subset | 0 for every column |
| dataset2 series in subset | 15, subset span `2019-02-01T11:10 … 2019-06-20T10:00` |
| dataset2 missing fractions in subset | 6.5 % (`CPU_Load`, `Network_RX/TX`) to 13.7 % (`CPU_cores_used`); power/temperature ≈11 % |
| dataset2 data-order caveat | 15 negative within-series deltas (−3000 s) and 90 duplicate `(series, time)` pairs in the bounded subset → file is not strictly chronological and needs sort/dedup review before any use |

**Aggregation feasibility verdict:** `not_defensible_without_assumptions`
(status `native_preserved_not_15min`, `fifteen_minute_product_created=false`).

Reasons (also stored in metadata/registry):

1. 10-minute samples do not tile a 15-minute interval (1.5 samples/bin); a plain mean or
   forward fill would fabricate the interval-average statistic instead of preserving it.
2. A time-weighted 10→15-minute resampling is only defensible if the file's timestamps are
   interval-start labels and the values are interval means; the local file documents neither
   units nor interval semantics, so that assumption is unaudited.
3. Series contain multi-day gaps; resampling would bridge holes that must stay visible.

Therefore UCC is **not** part of the 15-minute source panel. It stays a native 10-minute
candidate for a future Stage-3 pretraining path (native per-source frequencies are allowed by
`FeatureSpec.native_frequency` / `sampling_metadata`), after a full-file, unit/timezone audit.
Known limitations are stored in `source_metadata.json` (`limitations`, bounded audit only).
The 16 UCC field semantics are registered in the field registry with `unit: unit_unverified`
and `statistic: instantaneous`; they are not merged with M100 on name alone (alignment report §3).

## 8. Candidate sources deliberately not used

* **Alibaba GPU v2020** — registered as `candidate`, status `not_ingested`:
  * `pai_machine_metric.csv` in this local copy is **headerless** (first line begins with a
    machine-id hash; no column names), so metric columns cannot be mapped to audited fields
    without the official schema.
  * the other tables are event/job/instance/sensor tables in long per-event form, not
    aggregated to continuous 15-minute series.
  * task book §3.3 requires keeping raw events + field notes and marking aggregation
    incomplete rather than manufacturing continuous data. No 15-minute series was fabricated.
  * file sizes were recorded (machine metric 437.4 MB, sensor table 1,056.9 MB, instance
    table 1,975.2 MB, job table 133.3 MB, task table 113.4 MB, group-tag table 115.5 MB,
    machine spec 73 kB); SHA256 was intentionally not computed for a non-ingested source.
* **Google PowerData 2019** — already registered in `configs/datasets/registry.yaml` as a
  utilisation panel (`power_utilization`, unitless). Utilisation cannot be converted to kW
  without an external capacity calibration, so it is not a Stage-1 source-panel candidate.
* **NLR Eagle** — metadata-only candidate in the existing registry; no local data was
  reviewed in this round, so it is not claimed as processed.
* **Synthetic datasets** — engineering-only (`synthetic_simulated`), never part of the real
  source domain and never used to produce the analysis above.

## 9. Training approval and honesty statement

`source_metadata.json → training_approval`:

```
approved_for_pretraining: false
approved_for_facility_kw_training: false
approved_for_stage1_analysis: true
reason: Stage 1 is analysis/preprocessing only. M100 and UCC audits are partial/unverified,
        so no source-domain dataset is approved for training in this round.
```

Per-source reasons are recorded (`sources.m100_exadata.approval_reason`,
`sources.ucc2023.approval_reason`, `sources.alibaba_gpu_v2020.approval_reason`). M100 field
units/statistics are inferred from vendor telemetry and column naming; the physical mapping
of the p0/p1/ps0/ps1 channels is not independently documented — that is why Stage-3 use needs
a separate field-semantics review. No claim is made that the auxiliary transfer path or the
operational-metric→power mapping works.

## 10. Stage-3 handover (task book §13.7)

What Stage 3 receives and how to use it:

1. **`processed_source_features.parquet`** — 17.2 M-row long panel with
   `(timestamp, domain_id, feature_name, feature_value, unit, observed_mask, coverage_ratio)`.
   Pivot per `domain_id` to obtain per-node 15-minute series at native frequency; use
   `observed_mask` (never zeros) for missingness; never resample across the preserved gaps.
2. **`source_metadata.json`** — per-field `semantic_type`, `aggregation_rule`,
   `source_field`, `source_dataset`, `availability`, `unit`, `physical_range`,
   `audit_status`, `scope`/measurement point, plus the native-frequency audit and the
   node-vs-facility statement. Stage 3 must build `PretrainBatch`-style inputs from these
   fields with `available_at = timestamp` (historical observations) and keep `F` per source.
3. **`configs/datasets/feature_registry.yaml`** — the 12 M100 source fields are registered as
   `source_only` relative to the NLR target, with reasons; no shared field exists yet
   (alignment report), so Stage 3 cannot instantiate a shared-representation transfer with
   the current target without additional target operational metrics.
4. **UCC** — 16 field definitions + native 10-minute audit are registered but **not**
   panelised. Stage 3 may use native UCC after a full audit; it must not be force-resampled
   to 15 minutes without an explicit interval-label/unit assumption.
5. **Not delivered on purpose:** no operational-metric forecasting model, no
   operational→power mapping, no fusion, no global fusion weight (remains `None`), no
   target-only field execution. Those are Stage 3/4/5 work.

## 11. Reproduce

```powershell
# full REAL source pipeline (writes panel + metadata + quality + tracked registry)
& .\.venv\Scripts\python.exe -m scripts.prepare_source_features --config configs/datasets/sources.yaml

# bounded quick check (two nodes, 288 native samples each)
& .\.venv\Scripts\python.exe -m scripts.prepare_source_features --nodes 10 11 --max-rows-per-node 288

# source characteristics figures + statistics
& .\.venv\Scripts\python.exe -m scripts.analyze_sources --config configs/datasets/sources.yaml

# tests
& .\.venv\Scripts\python.exe -m pytest -q tests/test_source_features.py tests/test_feature_alignment.py
```

## 12. Deviations / limitations (honest list)

* UCC audit is **bounded** (first 300,000 rows per file); full-file span, series count, units
  and timezone remain unverified. UCC is not approved for training.
* M100's `p0/p1/ps0/ps1` channel semantics and GPU presence per node are inferred from
  vendor telemetry and column naming, not independently documented.
* M100 is node-level: nothing here supports a facility-level operational predictor.
* The 12 curated `_avg` metrics exclude `_std/_min/_max` interval extremes, which may be
  useful later and can be registered with their own statistics.
* The example figures are one node/window; they illustrate real telemetry, not a
  representative sample of every node.
