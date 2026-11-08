# Data rules

Rules for every training file, from the pilot audit of 2026-11-05/07. Each rule names the evidence behind it and the
check that enforces it. A rule changes only when a G3 result says so.

## What went wrong in the 10k pilot

| Finding | Pilot data | What the eval needs |
|---|---|---|
| Databases are tiny, so "verified" means little | about 20 rows per database; 99.5% of gold answers have 2 rows or fewer; planted mistakes caught 19% | BIRD databases: about 222,000 rows |
| 35% of targets were never independently checked | SQLite rows: SynSQL's SQL is both the target and its own gold | |
| Teacher prompt leaked into the reasoning | 14.5% of rows talk about benchmarks, gold answers, "search memory" | |
| Wrong SQL skills (share of queries) | CAST 6.9%, CASE 2.5%, dates 0.5%, division 3.2%, LIKE 1.7%, window 16.4% | CAST 60%, CASE 42%, dates 35%, division 20%, LIKE 10%, window 5% |
| Names are too regular | 0% of columns have spaces, 3% are cryptic, the same names everywhere (`user_id`, `created_at`) | 5% with spaces, 16% cryptic (`a10`, `numtsttakr`), 7% of TEXT columns hold numbers |
| Hints are rare and of the wrong kind | 22% have one, only on vague or metaphorical questions, some say "None required" | 73% have one, short definitions ("X refers to Y") |
| Questions are casual and short | 39% start with "Could you / Hey"; median 161 characters | 0.6%; median 347 characters |
| Reasoning style is not the model's own | DeepSeek shorthand ("We need answer..."), median 1,685 characters | base Qwen: "The user wants... I need to: 1. ...", median 3,425 |
| Long traces dominate the loss | the longest 10% of examples hold 36% of the target tokens | |

The eval numbers above describe BIRD Mini-Dev only to show the gap. Never copy from it: rule 10.

## Rules

**1. Real databases with real data.** Train on databases large enough that a wrong query rarely matches by luck: BIRD
train and Spider train first; synthetic sources only with rule 2. *Check:* `python -m t2sql.stress` must catch at
least half of the planted mistakes on the original data.

**2. Every target is checked by something other than itself.** A target counts as verified only if it matches an
independent gold on the original data and on enlarged copies (`t2sql/stress.py`), or matches a second, independent
answer. No self-graded rows like the pilot's SQLite rows. *Check:* each row carries `verified_by`.

**3. Teach the skills the task needs.** The share of CAST, CASE, date functions, division, LIKE, subqueries and
window functions in the targets should be within 2x of a reference built from BIRD train and Spider train (never from
eval sets). *Check:* feature table per training file, next to the reference.

**4. English names, with real-world mess.** Table and column names are English. Keep the messiness real databases
have: names with spaces or capitals that need quotes, cryptic abbreviations, numbers and dates stored as TEXT.
Transliterated, Uzbek, Cyrillic and mixed-script names are out of v1 (decided 2026-11-05).

**5. Hints like the real thing.** About 70% of questions carry a short English hint that defines a term, a formula
or a value ("eligible free rate = Free Meal Count / Enrollment"). No placeholder hints ("None required").

**6. Questions people would ask.** Mostly neutral wording; at most 10% casual and at most 5% metaphorical. Russian
and Uzbek versions keep quoted values verbatim; the hint stays in English.

**7. Clean reasoning.** No talk about benchmarks, gold answers, training, the prompt, or memory (`META_TALK` in
`t2sql/pilot.py` drops such rows). The teacher prompt never mentions training (`TEACHER_SYSTEM`). Prefer the base
model's own reasoning style; keep each trace under about 4,000 tokens, or weight each example equally in the loss, so
that long rambling traces do not dominate.

**8. Languages stay.** Questions in English, Russian, Uzbek Latin and Uzbek Cyrillic, as in the language split. Answers
and reasoning in English.

**9. One change at a time, at the same size.** Compare two data versions only at the same number of questions and the
same training settings. A result from a different size says nothing about the change (the 500-question run).

**10. Eval sets are for measuring only.** Never train on, copy from, or tune hints to Spider dev, BIRD dev and Mini-Dev,
PAUQ test, RedSQL or BIRD-CRITIC. The leakage check runs on every training file.

## Training rules

- Learning rate 5e-6 or lower for full fine-tuning: 2e-5 raised syntax errors from 1.0% to 11.0% after 48 steps.
- A run is damaging the model if syntax errors on G3 rise more than 1 point above base (1.0%): stop and lower the rate.
- The round trip is checked (2026-11-07): base saved through `modal/train.py` with learning rate 0 scores like base
  (15.5 vs 16.1, p=.51), so a drop after training is a training effect, not a saving or loading bug.
- Do not train on teacher reasoning for now. Every model trained on DeepSeek's traces (SynSQL or BIRD train, 500 or
  5,000 questions, 2e-5 or 5e-6) thought less than base (mean 1,240-1,864 tokens vs 3,976) and scored lower (6.0-12.8
  vs 16.1). Real BIRD data did help against SynSQL (10.5 vs 8.8; unknown columns 16.5% -> 8.1%), so keep rules 1-6.
- BIRD train is much easier than the eval (Mini-Dev is 92% "challenging"; subqueries 62% vs 8%), and keeping only
  answers that match gold halves the hard skills again (CAST 8.9% -> 5.4%). Check the skill mix after filtering.

## How a result is read (G3)

Same 1,100 prompts (`data/eval/baseline_prompts.jsonl`), thinking on, 12,000-token limit, base recomputed at the same
limit. Report per language, won/lost against base with a McNemar p-value, and the error types (no SQL, unknown column,
syntax, type, GROUP BY). The bar: better than base in every language.
