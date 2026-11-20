# Model Monitoring & Retraining Workflow: Explanation and Evidence

Project: Respiratory Risk Alert (forecasts tomorrow's Low / Moderate / High risk class, a rule-based proxy, not a clinical outcome).
Entry point: `python -m src.monitor` (code in `src/monitor.py`, tests in `tests/test_monitor.py`).

## 1. Scenario
The model is deployed (`models/champion/`). New labelled days keep arriving, so we must check the champion is still reliable,
retrain on the enlarged data and decide, with a recorded rule, whether the new model may replace it.

The "incoming batch" is `data/incoming/new_batch.csv`: the newest 60 days (2026-09-21 to 2026-11-19),
created with `python -m src.monitor make-batch`. These days were after the champion's training end (2026-09-19), so it had not seen them.
Anything older than the batch's first date counts as "existing training data".

## 2. Workflow

| Task | Implementation | Evidence file |
|---|---|---|
| A. Prepare new data | Checks the required columns (`date` + 10 base columns), counts missing values per column, checks duplicate IDs (the record ID is the date) and overlap with existing history, prints size and target distribution (label computed with the project's own rule). | `reports/data_validation.json` |
| B. Evaluate existing model | Loads `models/champion/model.joblib`, predicts on the labelled batch, computes Accuracy, Precision, Recall, F1 (macro), per-class scores and the confusion matrix. | `reports/evaluation_report.json`, `.md` |
| C. Retrain | Existing history + batch, built with the unchanged `build_dataset` pipeline. Model type is picked by time-ordered CV (same candidates as `train.py`). Saved separately. Logged to MLflow when installed (experiment `respiratory-risk-monitoring`). | `models/candidate/` |
| D. Quality gate | Rule below. Decision is written to a report and appended to a log. The champion is never overwritten by this step. | `reports/quality_gate.json`, `reports/decision_log.jsonl` |

**Leakage control.** The newest 30 batch days (`GATE_DAYS`) are held out of retraining. The champion and the candidate are both scored on
that same unseen window (2026-10-20 to 2026-11-18), so the comparison is fair. The candidate trains on everything before it.

## 3. Quality gate and metric justification
The candidate is **eligible for promotion** only if both hold on the gate window:
1. Macro-F1 of the candidate is greater than the champion's macro-F1 (margin `MIN_IMPROVEMENT`, default 0).
2. Recall of the "High" class is not more than 0.05 below the champion's.

Otherwise the decision is `RETAIN_CHAMPION`.

**Why macro-F1?** The classes are imbalanced, and plain accuracy lets a model do well by ignoring the rare High class. Macro-F1 weights every class
equally and balances precision and recall. It is also the metric the daily-retrain gate in `train.py` uses, so decisions stay consistent.
**Why the High-recall guard?** Missing a high-risk day is the costliest mistake for this application, so a model must not buy a higher F1 by missing more of them.

Promotion is a separate, explicit, logged step: `python -m src.monitor promote` (refuses if not eligible, archives the old champion to `models/archive/`).

## 4. Execution evidence (real run)
```
[A] batch validated: passed_with_warnings; 60 rows 2026-09-21..2026-11-19; target distribution Low=0 (0%), Moderate=24 (41%), High=35 (59%)
    warning: class(es) absent from batch: ['Low']; macro metrics average over classes that occur
[B] champion on new batch (n=59): acc=0.763 P=0.759 R=0.767 F1=0.759
[C] candidate 'hist_gradient_boosting' trained on 302 rows (through 2026-10-19) -> models/candidate/model.joblib
[monitor] mlflow not installed; skipping experiment tracking
[D] RETAIN_CHAMPION
    criterion 1 (macro-F1 0.659 vs champion 0.713, required margin > 0.0): FAIL
    criterion 2 (High-class recall 0.818 vs champion 0.818, max allowed drop 0.05): PASS
    champion NOT modified.
```

**A. Data validation:** status `passed_with_warnings`; 60 rows; missing values: none; duplicate IDs: 0;
target distribution Low=0 (0%), Moderate=24 (41%), High=35 (59%).
The batch is an autumn/winter pollution period, so no "Low" days occurred; macro metrics therefore average over the classes that occur.

**B. Champion on the new batch (n=59):**

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|
| 0.763 | 0.759 | 0.767 | 0.759 |

**C. Candidate:** `models/candidate/model.joblib` (model type and training window are in `models/candidate/metadata.json`).

**D. Gate on the unseen window (n=30):**

| Model | Macro-F1 | Accuracy | High recall |
|---|---|---|---|
| Champion | 0.713 | 0.767 | 0.818 |
| Candidate | 0.659 | 0.733 | 0.818 |

Decision: **RETAIN_CHAMPION**. criterion 1 (macro-F1 0.659 vs champion 0.713, required margin > 0.0): FAIL criterion 2 (High-class recall 0.818 vs champion 0.818, max allowed drop 0.05): PASS The champion file is byte-identical before and after the run.

## 5. How to reproduce
```bash
pip install -r requirements.txt
python -m src.monitor make-batch
python -m src.monitor
pytest -q
```
Optional: `mlflow ui` to browse the tracked run (needs `mlflow`; the workflow skips tracking silently if it is not installed. It was not installed in the environment where the evidence above was produced).

## 6. Limitations
- The 30-day gate window is small, so the decision is noisy; raise `GATE_DAYS` / `MIN_IMPROVEMENT` for stricter gating.
- The label is a rule-based proxy for respiratory risk, not clinical data.
- Retraining excludes the gate window, so a promoted candidate does not use the very latest 30 days (they can be added in the next cycle).
