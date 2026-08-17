---
language:
  - ko
  - en
license:
  - cc-by-4.0
  - apache-2.0
pretty_name: "JAMO Gold — Human-Validated Hangul Rendering Judgments"
task_categories:
  - image-to-text
  - text-to-image
tags:
  - ocr
  - evaluation
  - benchmark
  - text-rendering
  - korean
  - hangul
  - measurement-validity
  - human-annotation
  - vision-language
  - multimodal
size_categories:
  - n<1K
configs:
  - config_name: gold_pilot
    data_files:
      - split: train
        path: data/gold_pilot/*.parquet
  - config_name: synthetic_malformed
    data_files:
      - split: train
        path: data/synthetic_malformed/*.parquet
---

# JAMO Gold — Human-Validated Hangul Rendering Judgments

![A generated glyph resembling 田 in place of a valid Hangul coda, alongside a correctly formed output for the same target](figures/fig1_problem_example.png)

A small, carefully labelled dataset for a specific purpose: **testing whether
an OCR engine can tell that a glyph is not a real character.**

It is *not* a Hangul OCR training set, and *not* a model leaderboard.

---

## Why this exists

Benchmarks that measure visual text rendering score generated images with
OCR. But OCR engines are closed-set classifiers — given an image, they must
return *some* character from their inventory. They cannot say "this isn't a
character at all".

Generative models fail in exactly that way. In this dataset, **21.7% of
generated images are Hangul-shaped glyphs that cannot be typed** — for
example, a final consonant ㅁ drawn with an internal cross (making it 田).
Three OCR engines each silently map 77–83% of these onto a valid syllable.

This dataset provides the human labels needed to measure that gap.

Full analysis: [`docs/TECHNICAL_NOTE.md`](docs/TECHNICAL_NOTE.md).

---

## Quickstart

```python
from datasets import load_dataset

# 300 real generations, human-validated labels + 3 OCR engines' readings (no images)
gold = load_dataset("Nasser4963/jamo-gold", "gold_pilot", split="train")

# 581 font renderings with programmatic perturbations, ground truth by construction (images included)
synth = load_dataset("Nasser4963/jamo-gold", "synthetic_malformed", split="train")

# Concealment rate: how often does a judge return SOME valid syllable
# for a glyph that, by construction, isn't one?
invalid = synth.filter(lambda r: not r["is_control"])
concealed = sum(r["easyocr_reading"] is not None for r in invalid)
print(f"{concealed}/{len(invalid)} = {concealed / len(invalid):.1%}")
```

---

## Dataset structure

### `gold_pilot` — 300 generated images with human labels

Real outputs from a commercial text-to-image model accessed via a paid API,
single Hangul syllable targets, 18-cell stratified design. We withhold the
vendor and model identifier pending written clarification on redistribution
terms (see [`docs/RELEASE.md`](docs/RELEASE.md)); this is a disclosure-scope
choice, not an attempt to obscure the source, and the released
`image_manifest.jsonl` (prompts + hashes, no pixels) already documents
everything needed to regenerate the set once terms are confirmed.

**The raw images (`image` field) are not included in this split.** The
released `data/gold_pilot/*.parquet` (loads automatically via `load_dataset`,
also powers the Dataset Viewer preview on this page) ships the label and
judge fields only, plus `image_manifest.jsonl` for regeneration once terms
are confirmed.

| Field | Type | Description |
|---|---|---|
| `image_id` | string | Stable hash ID (no target leakage); joins to `image_manifest.jsonl` |
| `target` | string | The syllable the model was asked to draw |
| `template_id` | string | `T1` / `T2` (prompt template) |
| `n_sample_idx` | int | Sample index within the 18-cell design |
| `prompt` | string | Full prompt text sent to the model |
| `human_valid` | bool | **Primary label.** Is this a typeable syllable? |
| `human_reading` | string \| null | Transcription; present only when `human_valid` |
| `vowel_class` | string | `simple_V` / `complex_V` |
| `coda_class` | string | `no_T` / `simple_T` / `cluster_T` |
| `onset_group` | string | `simple_O` / `aspir_O` / `tense_O` |
| `clova_reading` | string \| null | Naver CLOVA General OCR output |
| `easyocr_reading` | string \| null | EasyOCR output |
| `easyocr_confidence` | float | EasyOCR confidence, 0–1 |
| `paddleocr_reading` | string \| null | PaddleOCR (`korean_PP-OCRv5`) output |
| `paddleocr_confidence` | float | PaddleOCR confidence, 0–1 |

The 4-way subcategory (`malformed` / `non_hangul` / `multi_syllable`) is
**not** in this table — per the reliability numbers below (α = 0.576), it is
not reliable enough to ship as a column. Use `human_valid`.

### `synthetic_malformed` — 581 controlled items, no generative model

Font renderings, programmatically perturbed. **Ground truth is guaranteed by
construction.** Use this split to test a judge without depending on any
generator. Unlike `gold_pilot`, images **are** included — SIL OFL font
derivatives carry no redistribution restriction — so this split has full
Dataset Viewer thumbnail previews.

| Field | Type | Description |
|---|---|---|
| `image` | image | Rendered glyph, 1024×1024 |
| `char` | string | Base syllable before perturbation |
| `kind` | string | `control` / `add_stroke` / `remove_stroke` |
| `severity` | string | `none` / `mild` / `moderate` / `severe` |
| `is_control` | bool | If true, the glyph is a real, unmodified syllable |
| `font_name` | string | OFL font used |
| `tm_reading` | string | template_match (closed-form) reading — always answers |
| `tm_score` | float | template_match similarity score |
| `tm_matches_original` | bool | Did template_match snap back to `char`? |
| `easyocr_reading` | string \| null | EasyOCR output (open-form — can abstain) |
| `easyocr_confidence` | float | EasyOCR confidence, 0–1 |
| `easyocr_matches_original` | bool | Did EasyOCR read `char` exactly? |

`is_control == False` ⟹ the glyph is **not** a valid syllable. Any judge
returning a confident valid-syllable reading on those items is concealing.
Reproduction: `python scripts/build_hf_dataset.py` regenerates these images
deterministically (fixed seed) from `jamo_bench.synthetic_malformed`, then
asserts row-for-row alignment against the archived eval results before
writing — see the script docstring for exact parameters.

---

## The labelling criterion

The primary label answers one question:

> **Can this glyph be typed on a standard keyboard?**
> (Equivalently: is it one of the 11,172 precomposed Hangul syllables?)

This replaced a subjective "legible / illegible" criterion after a pilot
showed the latter was applied inconsistently. It is binary, checkable, and
reproducible.

### Reliability (two independent annotators, 87-item overlap)

| Judgment | Agreement |
|---|---|
| **Binary validity** | 97.7% raw · **Krippendorff's α = 0.942** |
| Transcription, given both said valid | **23/23 = 100%** |
| 4-way subcategory | α = 0.576 |

**Use the binary label.** The subcategories (`malformed` vs `non_hangul`) are
provided for exploration but are **not reliable** — the boundary is a
continuum, and 17 of 21 inter-annotator disagreements fall on it.

---

## Intended uses

✅ **Testing whether a judge can abstain.** Feed it `synthetic_malformed`;
measure how often it returns a valid syllable on `is_control == False` items,
and whether its confidence distinguishes them. (In our tests: it does not —
Mann–Whitney p = 0.841.)

✅ **Calibrating an OCR-based scorer.** Compare your engine's readings to
`human_reading` on `gold_pilot` to estimate your own measurement bias before
publishing benchmark numbers.

✅ **Replicating the concealment finding** with a fourth engine.

## Uses to avoid

❌ **Training a Hangul OCR model.** 300 images, heavily skewed toward rare
syllables. It will not generalise.

❌ **Ranking generative models.** One generator, no comparison set. The
generator here is an instance used to obtain realistic failures, not a
subject of evaluation.

❌ **Citing the generator's accuracy as a model capability measure.** With
n = 300 on a deliberately rare-syllable-weighted design, the number is not a
general capability estimate.

❌ **Using the 4-way subcategory as ground truth** (α = 0.576).

---

## Known limitations

- **Single generator, n = 300.** The 21.7% invalid rate is measured on one
  model; we do not claim it generalises. The *judge behaviour* claim is
  supported by the generator-free `synthetic_malformed` split, which does
  generalise.
- **One of the two annotators is the dataset author.** The second is
  independent; all agreement statistics are between the two.
- **Per-cell values are unreliable.** ~12 images per 18-cell bucket. Only the
  three marginal axes (vowel, coda, onset) support inference.
- **Observational for structure claims.** "Complex codas are harder" is
  confounded with character frequency and stroke count; we make no causal
  claim. See §7 of the technical note.
- **`synthetic_malformed` perturbations are synthetic**, mirroring observed
  failure types but not sampled from a generator's true error distribution.

---

## Headline numbers

Recompute all of these with `python scripts/verify_claims.py`.

| Finding | Value |
|---|---|
| Outputs that are not valid syllables | **21.7%** (65/300) |
| Concealment rate — CLOVA / EasyOCR / PaddleOCR | **76.9% / 83.1% / 78.5%** |
| Engines snapping to *different* characters | 34/44 = **77.3%** |
| Accuracy understatement (human − OCR) | **+25.6 to +50.0pp** |
| Complex-coda penalty exaggeration | **2.3× – 4.0×** |
| Controlled test: confidence distinguishes fabricated glyphs? | **No** (p = 0.841) |

---

## Licensing

- **Annotations, code, metadata**: CC BY 4.0 / Apache 2.0
- **`synthetic_malformed` images**: derived from SIL OFL fonts (Noto Sans KR,
  Noto Serif KR, Pretendard); OFL notices included in `fonts/`
- **`gold_pilot` images**: generated by a commercial T2I API. Redistribution
  is subject to that provider's terms, **which we have not yet cleared**. The
  `image` field is therefore **not included** for any row in the public
  release; the repository ships the labels, all judge readings, and
  `image_manifest.jsonl` (prompts + hashes) so the set can be regenerated
  once terms are confirmed. Crops of three distinct images from this set
  appear across two illustrative figures in the technical note (not as
  dataset rows); see
  [`docs/RELEASE.md`](docs/RELEASE.md) for the scope and licensing of that
  exception, made at the depositor's discretion pending the same
  clarification.
- **If the images are released later**, they will **not** be under CC BY 4.0.
  A provider terms review (2026-08-12, not legal advice) found clauses in the
  underlying service's terms restricting use of model outputs for training,
  fine-tuning, annotation, or development of models or algorithms — CC BY 4.0
  does not exclude such use, so it is not an appropriate license for these
  images. If released, they will carry a restricted research/evaluation
  license: reproduction and evaluation permitted; training, fine-tuning, or
  algorithm development using the images prohibited; no removal of
  AI-generated content markers; no implied vendor endorsement. See
  [`docs/RELEASE.md`](docs/RELEASE.md) for the full policy and
  [`docs/BYTEPLUS_INQUIRY_DRAFT.md`](docs/BYTEPLUS_INQUIRY_DRAFT.md) for the
  pending clarification request.

## Citation

```bibtex
@techreport{lim2026jamo,
  title  = {OCR-Based Scoring Conceals Rendering Failures in
            Text-to-Image Evaluation: Evidence from Hangul},
  author = {Lim, Taewoo},
  year   = {2026},
  type   = {Technical Note}
}
```
