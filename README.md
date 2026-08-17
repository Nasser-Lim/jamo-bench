# JAMO

### Judging Accuracy of Machine-rendered Orthography

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21971468.svg)](https://doi.org/10.5281/zenodo.21971468)

An OCR-mediated benchmark that diagnoses Hangul visual-text rendering at the
jamo level (onset/nucleus/coda). The deliverable is not images — it is a
**prompt set + scorer + structural error analysis**. Full design background
is in [`JAMO_benchmark_design.md`](JAMO_benchmark_design.md) (v5) and
[`JAMO_v51_patch.md`](JAMO_v51_patch.md); the **actual v1 scope and why it
was cut down** is in [`docs/SCOPE.md`](docs/SCOPE.md); the full history of
what was built and what broke (and why) is in
[`docs/PROGRESS.md`](docs/PROGRESS.md). (Those two docs are in Korean —
the technical note below has an English original.)

> v1 implements a **narrowed subset** of the v5/v5.1 design, not the full
> scope — a single model (Seedream), 18-cell structural diagnosis, and
> quantifying how a judge silently snaps a nonexistent character to the
> nearest valid syllable. Cross-model comparison, Forge, a leaderboard, and
> the private-test holdout are not part of v1.
>
> **This project started as a Hangul-typography generation benchmark.** It
> pivoted into a measurement-validity study once it became clear the
> automated judge itself couldn't be trusted — see the four turning points
> in `docs/SCOPE.md`'s
> ["Why this project pivoted"](docs/SCOPE.md#프로젝트-전환-경위--한글-타이포그래피-벤치마크에서-측정-타당성-연구로)
> section.

## Data + Paper

| Where | What |
|---|---|
| [huggingface.co/datasets/Nasser4963/jamo-gold](https://huggingface.co/datasets/Nasser4963/jamo-gold) | Released data: human labels, judge outputs, prompt/hash manifests (generated images withheld, see [License](#license)) |
| [doi.org/10.5281/zenodo.21971468](https://doi.org/10.5281/zenodo.21971468) | Citable, DOI-pinned Zenodo archive of this repository's `v1.0.0` release |
| [`docs/TECHNICAL_NOTE.md`](docs/TECHNICAL_NOTE.md) | Technical note — claims, experiments, limitations. Korean translation: [`docs/TECHNICAL_NOTE.ko.md`](docs/TECHNICAL_NOTE.ko.md) |
| [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) | Dataset card (mirrors the HuggingFace card) |
| [`docs/SCOPE.md`](docs/SCOPE.md) | v1 scope decision — evidence tiers, what was included/excluded and why |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | 20-step build history, including rejected claims |

This repository is the **code companion** to the HuggingFace dataset above:
every number published there (and in the technical note) is reproducible
from this repo with zero API keys or original images required.

```bash
git clone https://github.com/Nasser-Lim/jamo-bench && cd jamo-bench
pip install -e ".[dev]"
python scripts/build_release.py   # builds the anonymized release/ bundle (0.9MB)
python scripts/verify_claims.py   # recomputes every number in the note from release/ alone
```

`verify_claims.py` reads only `release/` — **no API keys, no original
images** — recomputes every claim in the technical note, and exits non-zero
on any mismatch.

---

## 5-minute Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows
pip install -e ".[dev]"
pytest -q                         # 166 tests, no API keys needed
```

```python
from jamo_bench.decompose import decompose
from jamo_bench.score import score
from jamo_bench.partitioning import partition
from jamo_bench.judging_protocol import route, resolve_human

# Jamo decomposition
s = decompose("읽")
print(s.onset, s.nucleus, s.coda)          # ㅇ ㅣ ㄺ
print(s.coda_class_4)                       # cluster_mixed

# Scoring (target vs. OCR/judge candidate)
result = score("읽", "익")
print(result.verdict, result.coda_ok)       # VALID False

# 18-cell stratified sample (Core 540 / Cross-shared 250 / Cross-Struct 180)
p = partition(seed=42)
print(len(p.core_540), len(p.cross_shared_250))   # 540 250

# v1 judging protocol — CLOVA confidence>=0.80 auto-accepts, else routed to a human
d = route(clova_reading="갑", confidence=0.42, coda_class_3="simple_T")
print(d.needs_human, d.expected_bias_pp)    # True -0.9
d = resolve_human(d, human_valid=True, human_transcription="값")
print(d.reading)                            # 값

# Tense-consonant onsets go to a human even at high confidence
# (CLOVA is confidently wrong in exactly this region)
print(route("쌈", 0.99, "simple_T", onset_group="tense_O").needs_human)   # True
```

Features that need external services (image generation, OCR, human audit
UIs) require a `.env` file (see [Environment variables](#environment-variables)
below). Everything else — decomposition, scoring, partitioning, the judging
protocol — runs and tests entirely offline.

---

## Current status, in one paragraph

The core library and external integrations (BytePlus ModelArk/Seedream,
Naver CLOVA OCR) work. **CLOVA alone is not a valid primary judge** — 32.8%
agreement with human ground truth (`docs/PROGRESS.md`, steps 6–7). Instead,
the empirically validated **v1 judging protocol** auto-accepts CLOVA's
reading when `inferConfidence >= 0.80`, and routes everything else to a
human (`jamo_bench/judging_protocol.py`). Counter-intuitively, CLOVA loses
confidence precisely where it's untrustworthy (complex codas), so gating on
confidence alone makes the residual bias nearly flat across structural axes
(±1.6 percentage points). Two other approaches were tried and dropped:
coda-type routing between CLOVA and template_match (`hybrid_judge.py` —
more complex, no better), and connected-component-based OVERGEN
self-detection (`overgen.py` — capped at 15% detection because merged
multi-character glyphs can't be segmented). Both remain in the codebase but
are excluded from the v1 pipeline.

**Core finding:** judges (both CLOVA and template_match) conceal generation
failures by snapping a nonexistent glyph to the nearest valid syllable — a
full human audit of the 300-image pilot found 21.7% were not valid
completed syllables (malformed glyphs dominate, 14.0%). Humans' **binary**
judgment (valid completed syllable, yes/no) is highly reliable — Krippendorff
α=0.942 across 2 annotators × 87 items — while finer 4–5 category labels
(malformed/non_hangul/multi-syllable) are not (α=0.52–0.58) and are kept as
exploratory only.

**What v1 actually claims:** not "this model is X% accurate," but "how much
does the judge conceal the model's most common failure" and "structural
comparisons (complex coda vs. simple coda, etc.) on the human-verified Gold
split." Full history in `docs/PROGRESS.md` steps 6–15; scope rationale in
`docs/SCOPE.md`.

---

## Project structure

```
JAMO/
├── jamo_bench/                 core library (mostly testable with no API keys)
│   ├── decompose.py            jamo decomposition, 18-cell classification, coda_class_4/vowel_shape metadata
│   ├── score.py                VALID/OVERGEN/EMPTY/NON_HANGUL/HALLUCINATED scoring
│   ├── align.py                 multi-syllable (Word) edit-distance alignment, OVERGEN partial credit
│   ├── route.py                 Route A0/A1/B/C classifier
│   ├── metrics.py               3 chance baselines, bootstrap CI, confusion matrices
│   ├── partitioning.py          Core 540 / Cross-shared 250 / Cross-Struct 180
│   ├── prompts.py                PROMPT_SPECS.yaml loader (T1/T2/T3 templates)
│   ├── forge_render.py          OFL font clean/degraded renderer
│   ├── vision_heuristics.py     OCR-independent ink detection (has_ink_marks)
│   ├── judge_preprocess.py      text-area normalization before OCR
│   ├── match_region.py          F0–F4 region-matching fallback ladder
│   ├── judging_protocol.py      v1 judging protocol — CLOVA confidence gate + human ← current judge
│   ├── judge_ceiling.py         Judge Ceiling (clean/degraded) measurement
│   ├── modelark.py              BytePlus ModelArk (Seedream) image generation client
│   ├── clova_ocr.py             Naver CLOVA General OCR client
│   ├── audit_queue.py           human audit item queue builders (font-render / real images / Phase 2 relabel)
│   ├── template_match.py        closed-form shape matching (soft-IoU) — excluded from v1 pipeline, kept in code
│   ├── hybrid_judge.py          coda-type CLOVA/template_match routing — excluded from v1 (replaced by the confidence gate)
│   ├── overgen.py               connected-component OVERGEN self-detection — dropped, 15% detection ceiling, kept in code
│   └── vlm_judge.py             VLM judge candidate — dropped, cost/latency impractical, kept in code
│
├── scripts/                     runnable scripts (all lock-file protected against duplicate runs)
│   ├── run_pilot.py              pilot/main batch image generation + scoring runner
│   ├── rescore_pilot.py          re-score saved images (zero regeneration cost)
│   ├── measure_ceiling.py        18-cell Judge Ceiling measurement (resumable per cell)
│   ├── eval_confidence_gate.py   v1 judging protocol threshold sweep (produces the numbers above)
│   ├── eval_wellformedness.py    well-formedness signal (unexplained_ink, etc.) AUC evaluation
│   ├── eval_overgen.py           OVERGEN calibration — reproduces the negative result
│   ├── audit_server.py + audit_ui.html                human audit web UI — font renders (port 8877)
│   ├── audit_server_pilot.py + audit_ui_pilot.html     human audit web UI — real generations, round 1 (port 8878)
│   └── audit_server_pilot2.py + audit_ui_pilot2.html   human audit web UI — Phase 2 relabel / 2nd annotator (port 8879)
│
├── tests/                       166 tests, `pytest -q`
├── fonts/                       Noto Sans/Serif KR, Pretendard (all OFL, attribution files included)
├── results/                     empirical outputs (pilot images/scores, Ceiling, audit logs)
├── docs/
│   ├── SCOPE.md                  v1 scope decision — evidence tiers, inclusion/exclusion, trade-offs
│   ├── PROGRESS.md               step-by-step build history — what was built, what broke, why
│   └── partitioning.md           single-spec doc for data partition execution order
├── JAMO_benchmark_design.md     design spec v5 (v1 scope is narrowed by SCOPE.md)
├── JAMO_v51_patch.md            design patch v5.1
├── PROMPT_SPECS.yaml            T1/T2/T3 prompt templates (version-frozen)
├── jamo_*.py (root)             one-off design-verification scripts (early arithmetic checks, for reference)
└── .env                         API keys (gitignored)
```

---

## Judging protocol (v1)

**Full rule:**

```
confidence >= 0.80 (CLOVA inferConfidence)
  AND onset_group != tense_O (ㄲㄸㅃㅆㅉ)   →  accept CLOVA's reading as-is
otherwise (including detection failure)      →  route to a human
                                                  1) is it a valid completed syllable? (binary, α=0.942)
                                                  2) if valid, transcribe it
```

Why `tense_O` is excluded: confidence's self-correction breaks down only in
this region — CLOVA↔human disagreement in the auto-accept band is **47.4%**
for tense onsets (vs. 4.5% for simple, 21.7% for aspirated). CLOVA is
**confidently wrong** on tense-consonant-onset syllables
(`docs/PROGRESS.md`, step 16).

| Metric | Value | Source |
|---|---|---|
| Human-queue rate | 50.0% | `scripts/eval_confidence_gate.py` sweep, n=256 |
| Overall bias | −0.8pp | always at or below human ground truth (sign-consistent) |
| Per-axis bias spread | coda 1.5 / vowel 0.5 / onset-group 1.9pp | passes ≤2.5pp on all three of the 18-cell axes |
| Coda-gap reconstruction error | −1.6pp | vs. −15.8pp on the human ground truth |
| Jamo-position error slope overstatement | +3.3–3.9pp | ordering preserved; absolute values trusted only on the Gold split |
| Concealment (invalid glyph scored as correct) | 0 of 133 auto-accepted images | n small — interpret as a 95% upper bound of ≈2.2% |

**Non-negotiable when reporting results — always report
`jamo_bench.judging_protocol.MEASURED_BIAS_PP` alongside any number.**
Absolute accuracy claims (e.g. "this model is X% accurate") fall short of
the design's own scoring floor (85% human agreement, design §14.2), so they
are made **only on the fully human-verified Gold split**; any number that
includes auto-accepted judgments is used **only for structural comparisons**
(e.g. "complex codas score Ypp worse than simple codas").

Judges that were tried and excluded from v1:

| Judge | Status |
|---|---|
| CLOVA alone | disqualified — 32.8% agreement with human ground truth (180-item gold set) |
| VLM (ModelArk) | shelved — up to 5 minutes per image from reasoning-token cost |
| template_match | kept in code, excluded from v1 — the confidence gate is simpler and has lower bias spread |
| hybrid_judge (coda-type routing) | kept in code, excluded from v1 — 2.5pp bias spread, worse than the confidence gate's 1.6pp |
| overgen (connected-component) | kept in code, dropped — detection rate capped at 15% |

---

## Environment variables

Set these in a (gitignored) `.env`:

```bash
ARK_API_KEY=...              # BytePlus ModelArk API key
ARK_MODEL_SEEDREAM=ep-...    # Seedream image generation endpoint
ARK_MODEL_VLM=ep-...         # VLM judge endpoint (optional)
CLOVA_API_URL=...            # Naver CLOVA General OCR domain URL
CLOVA_SECRET_KEY=...         # Naver CLOVA secret key
```

Everything else in `jamo_bench` — decompose/score/align/route/metrics/
partitioning/forge_render/judging_protocol/template_match/hybrid_judge/
overgen — and its tests run fine with no keys at all. Only
`modelark.py`/`clova_ocr.py`/`vlm_judge.py` need keys, and only when they
actually make a network call.

---

## Representative commands

```bash
# See the run plan without generating images
python scripts/run_pilot.py --dry-run

# Re-score saved pilot images (only re-calls OCR, zero generation cost)
python scripts/rescore_pilot.py

# Measure the 18-cell Judge Ceiling (resumable per cell)
python scripts/measure_ceiling.py

# Recompute the v1 judging protocol threshold (0 API calls, reuses saved audit results)
python scripts/eval_confidence_gate.py

# Human audit — real generations, round 1, http://127.0.0.1:8878
python scripts/audit_server_pilot.py

# Human audit — Phase 2 (relabel / 2nd annotator), http://127.0.0.1:8879
python scripts/audit_server_pilot2.py
```

---

## How to cite

```bibtex
@software{lim2026jamo,
  author    = {Lim, Taewoo},
  title     = {JAMO: Judging Accuracy of Machine-rendered Orthography},
  version   = {v1.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21971468},
  url       = {https://doi.org/10.5281/zenodo.21971468}
}
```

See also [`CITATION.cff`](CITATION.cff).

## License

`jamo_bench/` code is Apache 2.0 / CC0 per design §16. Fonts in `fonts/`
are all SIL Open Font License (attribution files included). Release scope
for the generated benchmark images themselves follows each model's own
terms of service (§10.2) — currently withheld by default pending that
review; see [`docs/RELEASE.md`](docs/RELEASE.md).
