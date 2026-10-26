# Stage-1 Field-Alignment Report — M100 vs NLR target / M100 vs UCC

> **Deliverable:** `configs/datasets/feature_registry.yaml` (tracked) plus this report.
> **Implementation:** `src/data/source_features.py` uses `src.data.registry.FieldRegistry.align`
> directly — alignment is **not** reimplemented anywhere.
> **Honest headline:** with the current target domain there is **no defensible shared field**
> between M100 node telemetry and the NLR ESIF target. The registry returns
> `no_shared_features` and *does not fabricate* a shared representation. The target-only
> fields keep a Stage-4 `local_weight: 1.0` reservation; the **global fusion weight stays
> unresolved (`None`)** and nothing is mapped/executed in Stages 1–2.

## 1. Method and rules (task book §3.4, data contract §4)

`FieldRegistry.align(source_id, target_id)` classifies every field into `shared`,
`source_only`, `target_only` or `incomparable`. A field is `shared` only when it has the
same canonical semantic, the **same unit** and a **compatible interval statistic**; the raw
column name is irrelevant. Same-name fields with different units/statistics are
`incomparable` and are never merged. The registry also returns explicit
`blocked_reasons`, and `global_fusion_weight` is always `None`.

Two alignment runs are produced:

| key | source id | target id | purpose |
|---|---|---|---|
| `m100_vs_nlr` | `m100_exadata_pretrain` (`unlabeled_pretrain`, target = None) | `nlr_esif` (`target_supervised`, `facility_derived_kw`) | primary source vs target domain |
| `m100_vs_ucc` | `ucc2023` (`unlabeled_pretrain`) | `m100_exadata_node_label` (comparison-only M100 node-power label) | cross-source field comparability: UCC vs M100 |

Why the second run uses an M100 "target": `FieldRegistry.align` requires a non-unlabelled
domain on the target side. M100 genuinely contains measured **node-scope** power, so the
alignment spec `m100_exadata_node_label` declares it as an auxiliary node-level label
(`label_origin: measured`, `scope: node`). This spec exists only for field-semantic
comparison; it is never promoted to facility kW and is never used by the Stage-2 model A.
Direction therefore only relabels the result: for `m100_vs_ucc`, `source_only` means
UCC-only and `target_only` means M100-only; the shared/incomparable classification is
symmetric.

## 2. Result 1 — M100 ExaData vs NLR ESIF target (primary alignment)

**Status: `no_shared_features` / `is_transferable=false`.**

| relation | count | fields |
|---|---|---|
| `shared` | **0** | — (no shared representation is invented) |
| `source_only` | 12 | `node_total_power_w`, `cpu0_power_w`, `cpu1_power_w`, `cpu0_memory_power_w`, `cpu0_io_power_w`, `psu0_input_power_w`, `psu1_input_power_w`, `disk_power_w`, `node_ambient_temp_c`, `gpu0_core_temp_c`, `gpu1_core_temp_c`, `fan0_0_speed_rpm` |
| `target_only` | 2 | `site_outdoor_temperature_c`, `site_outdoor_humidity_pct` |
| `incomparable` | 0 | — |

Blocked reason stored in the registry:

> no comparable shared feature between `m100_exadata_pretrain` and `nlr_esif`; transfer is
> not available yet and no fake shared representation is fabricated.

Why every superficially similar pair is **not** shared (field-registry reasons):

* **Power is a different measurement quantity and scope.**
  M100 `node_total_power_w` / CPU/PSU power fields are **node-level hardware rails** (W,
  15-min interval means). The target label `facility_derived_kw` is a **facility-level
  component sum** of five sub-meters, and those five components are the **label
  composition**, not candidate inputs (task book §3.5). Node power must not be renamed or
  scaled into facility kW, so there is no power field to join — only an auxiliary node task.
* **Temperature is a different measurement point.**
  M100 `node_ambient_temp_c` is a chassis/inlet sensor inside/next to a server
  (`measurement_point=node chassis ambient/inlet`), while the target
  `site_outdoor_temperature_c` is the facility's **outdoor weather station**
  (`measurement_point=outdoor weather station`). They are not the same physical variable, so
  they are deliberately given different canonical names and classified source-only /
  target-only. A same-name merge would be a fabricated shared field.
* **The target carries no operational metrics.**
  The NLR target reference carries only the five power components (label composition) and
  outdoor weather. There is no CPU/GPU utilisation, task count, memory or IT-power feature
  that could serve as a shared operational input with M100. Consequently there is nothing
  for the auxiliary path to transfer onto yet.

Target-only reservations (semantic reservation only, **not executed**):

| target-only field | unit | statistic | `future_mapping.local_weight` | `global_fusion_weight` |
|---|---|---|---|---|
| `site_outdoor_temperature_c` | °C | interval_mean | **1.0** | **null (unresolved)** |
| `site_outdoor_humidity_pct` | % | interval_mean | **1.0** | **null (unresolved)** |

`local_weight = 1.0` means only "this field is not additionally suppressed inside its own
local branch" — it is **not** a physical regression coefficient, **not** a global fusion
share and **not** a prediction weight. The global fusion weight is a research decision that
this round does not take; the registry stores `global_fusion_weight: null` and the test
suite walks the whole document to prove no global weight of `1` (or any number) exists.

Target label composition (registered as label semantics, **not** features):
`facility_derived_kw = it_power_kw + cooling_kw + hvac_kw + plug_and_light_kw + pump_kw`
(`label_origin=component_sum`, label type `facility_derived_kw`). The five components have
`input_eligible: false`; they are **not** in the target feature list and must never be used
as future-known inputs for an "operational metric → power" mapping claim (task book §3.5).

The target reference `DatasetSpec` built here is documentation for alignment only:
**`target-power` owns the authoritative processed target files** (`data/processed/nlr*.parquet`).
Its documented facts (15-minute native, tz unverified-but-UTC-assumed, 343,687 processed rows,
2015-11-10 … 2025-08-29 span; label is a derived facility sum) are recorded in
`configs/datasets/sources.yaml → target_reference` and used unchanged.

## 3. Result 2 — M100 vs UCC 2023 (cross-source comparability)

**Status: `no_shared_features` / `is_transferable=false`.** The UCC fields are
`unit_unverified` and `instantaneous` in the local copy; nothing is merged on name alone.

| relation (call direction: UCC = source, M100 node label = target) | count | fields |
|---|---|---|
| `shared` | **0** | — |
| `source_only` (UCC-only) | 15 | `cpu_load`, `network_receive_rate`, `network_transmit_rate`, `node_inlet_air_temperature`, `cpu1_temperature`, `cpu2_temperature`, `fan1_speed`…`fan4_speed`, `ram_total`, `ram_used`, `running_vm_count`, `cpu_cores_total`, `cpu_cores_used` |
| `target_only` (M100-only) | 11 | all curated M100 fields except `node_total_power_w` |
| `incomparable` | **1** | `node_total_power_w` |

The single incomparable field is the important one: UCC `Power` is registered under the
same canonical hypothesis `node_total_power_w`, but the registry refuses to merge it:

> `node_total_power_w`: unit differs (`unit_unverified` vs `W`); values are not comparable
> without an explicit, audited conversion; statistic differs (`instantaneous` vs
> `interval_mean`); an interval mean and an instantaneous sample are different quantities.

Additionally `node_inlet_air_temperature` is kept separate from M100
`node_ambient_temp_c` (sensor position unverified) and from the target's outdoor weather —
no temperature field is merged across sources. UCC field units (W vs kW, RPM vs %, GiB,
bytes vs bytes/s) are undocumented in the local files; the registry records that uncertainty
explicitly instead of guessing, and UCC is kept at its native 10-minute frequency (see the
source report §7).

## 4. Tracked registry (`configs/datasets/feature_registry.yaml`)

The YAML is auto-generated by `scripts/prepare_source_features.py` from
`configs/datasets/sources.yaml` + the live `FieldRegistry.align` results, so it cannot drift
from the code; a test asserts the tracked file equals a freshly generated payload. Structure:

```text
schema_version, generated_by, stage, execution_status
fusion:            target_only_local_weight=1.0, global_fusion_weight=null, status=unresolved
approval:          Stage-1 analysis only; pretraining/facility-kW training not approved
sources:
  m100_exadata:    role, node scope/measurement boundary, audit_status, approval, 12 fields
  ucc2023:         role, native 10min, aggregation_status/verdict/reasons, 16 fields
  alibaba_gpu_v2020: candidate, status=not_ingested, reasons, fields=[]
  nlr_esif:        target-only features (2) with future_mapping.local_weight=1.0,
                   label (facility_derived_kw) + composition policy, owner=target-power
alignment:
  m100_exadata__vs__nlr_esif: status/counts/lists/reasons
  m100_exadata__vs__ucc2023:  status/counts/lists/reasons (+ direction note)
alignment_status_counts:      counts summary
```

Every field entry carries: canonical + raw name, unit, `semantic_type`, `statistic`,
`aggregation_rule`, `time_granularity`, `availability`, `future_known`, `physical_range`,
`missing_policy`, measurement point, scope, source dataset, `audit_status`, `feature_role`,
`role_context`, a non-empty `role_reason`, `future_mapping` (only for target-only fields) and
notes. Every relation (`source_only`/`target_only`/`incomparable`) in the alignment sections
has a non-empty reason.

## 5. Implications for Stage 3 (feature pretraining) and Stage 4 (mapping)

1. **Stage 3 can pretrain on M100 dynamics now.** The 12 M100 fields are a real, masked,
   native-15-minute multivariate panel across 16 node domains (17.2 M rows, 84 % observed
   overall, gaps preserved). Stage 3 can learn their evolution with per-source feature sets
   and per-domain sampling metadata; UCC can extend this at native 10-minute frequency after
   a full audit. No source is approved for training yet (partial/unverified audit), so this
   is a data-readiness statement, not a training result.
2. **Stage 3/4 transfer to the NLR target is currently blocked by field semantics.** With
   zero shared fields, there is no `shared` representation to transfer onto the target and no
   target operational input for the mapping stage. This is the honest outcome required by the
   task book: no same-dimension feature padding, no fabricated shared variables.
3. **What unblocks it (recommended next data actions):**
   * target-domain **operational metrics** with documented semantics/units — e.g. CPU/GPU
     utilisation, memory usage, task/concurrency counts, or IT power measured *independently*
     of the label composition — matching (or explicitly adaptable to) source fields;
   * a **full UCC audit** (units, timezone, full-file segments, sort/dedup fixes) so its
     `cpu_load`/`ram`/`fan` fields can become genuine `source_shared` candidates;
   * if no target operational metrics exist, restrict the auxiliary path to a separately
     declared proxy/shape task and never report it as a facility-kW result.
4. **Weather is not a future-known feature here.** The target weather fields are declared
   `historical_observed`; using them at forecast time would need an external weather forecast
   source with an auditable availability time. Stages 1–2 neither use them nor claim them as
   known-future.
5. **No mapping/fusion is executed.** No target-only field enters the Stage-2 main model A,
   no mapping branch is trained, and the global fusion weight remains `None` (unresolved by
   research decision, task book §1.2/§8.2).

## 6. Tests covering this report

`tests/test_feature_alignment.py` (bounded, fast) asserts, among others:

* M100 vs target has **no fabricated shared fields** (`shared=()`,
  `status=no_shared_features`, 12 source-only, 2 target-only, 0 incomparable);
* target-only fields carry `local_weight=1.0`, preserve raw names/units/domains, and the
  **global fusion weight is `None` everywhere** (whole-document walk);
* source-only/incomparable fields carry no future-mapping reservation;
* the same-name `node_total_power_w` (UCC vs M100) is **incomparable, not merged**, with
  explicit unit+statistic reasons, and is absent from `shared`/only lists;
* alignment status/reasons are explicit for every route;
* the tracked `feature_registry.yaml` loads and equals a freshly generated payload;
  `source_only`/`target_only`/`incomparable` reasons are non-empty; no
  `global_fusion_weight: 1` exists; label components are not registered as input features;
* the source `DatasetSpec` is `unlabeled_pretrain` with `target=None`, node-level and not
  approved for training; the target label classifies as `facility_derived_kw`.

Reproduce with:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q tests/test_feature_alignment.py tests/test_source_features.py
```
