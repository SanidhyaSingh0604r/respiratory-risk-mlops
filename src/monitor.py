"""Model-quality monitoring + retraining workflow (Tasks A-D).

    python -m src.monitor make-batch   # (demo) carve the newest days of daily.csv into an "incoming batch"
    python -m src.monitor              # A validate -> B evaluate champion -> C retrain candidate -> D quality gate
    python -m src.monitor promote      # explicit, logged promotion of an ELIGIBLE candidate

Design rules
  * The champion in models/champion/ is NEVER touched by the default run. The candidate is saved
    separately in models/candidate/ and only marked eligible / not eligible.
  * Every gate decision is written to reports/quality_gate.json and appended to
    reports/decision_log.jsonl.
  * The existing preprocessing pipeline (src.features.build_dataset) and model-selection code
    (src.train.candidate_models) are reused unchanged.
  * Leakage control: the newest GATE_DAYS of the batch are held out of retraining, and BOTH the
    champion and the candidate are scored on that same unseen window.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, precision_score, recall_score)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src import config
from src.features import build_dataset
from src.train import candidate_models, should_promote

REQUIRED_COLS = ["date"] + config.BASE_COLS
LABELS = [0, 1, 2]
HIGH = 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


# ---------------------------------------------------------------- demo helper
def make_batch(daily_path=config.DATA_PATH, batch_path=config.BATCH_PATH, days=config.BATCH_DAYS) -> pd.DataFrame:
    """Simulate 'new data arrived': the newest `days` rows of daily.csv become the incoming batch."""
    daily = pd.read_csv(daily_path)
    batch = daily.sort_values("date").tail(days)
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(batch_path, index=False)
    print(f"[monitor] wrote {len(batch)} rows ({batch['date'].min()} .. {batch['date'].max()}) -> {batch_path}")
    return batch


# ---------------------------------------------------------------- Task A
def validate_batch(batch: pd.DataFrame, existing: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate the incoming batch. Raises ValueError on blocking problems; returns a cleaned batch."""
    missing_cols = [c for c in REQUIRED_COLS if c not in batch.columns]
    if missing_cols:
        raise ValueError(f"batch is missing required columns: {missing_cols}")

    batch = batch.copy()
    batch["date"] = pd.to_datetime(batch["date"], errors="coerce")
    bad_dates = int(batch["date"].isna().sum())
    batch = batch.dropna(subset=["date"])
    if batch.empty:
        raise ValueError("batch has no rows with a valid date")

    missing_values = {c: int(n) for c, n in batch[config.BASE_COLS].isna().sum().items() if n}
    # "ID" of a record in this project = the calendar date (one row per day)
    dup_ids = int(batch["date"].duplicated().sum())
    existing_dates = set(pd.to_datetime(existing["date"]))
    overlap = int(batch["date"].isin(existing_dates).sum())
    batch = batch.sort_values("date").drop_duplicates("date", keep="last")

    # target distribution (classification) - computed with the project's own label rule, using
    # the existing history as context for lags / rolling windows
    ctx = pd.concat([existing.assign(date=pd.to_datetime(existing["date"])), batch], ignore_index=True)
    ctx = ctx.sort_values("date").drop_duplicates("date", keep="last")
    ds = build_dataset(ctx)
    ds = ds[(ds.index >= batch["date"].min()) & (ds.index <= batch["date"].max())]
    counts = ds["target"].value_counts().reindex(LABELS, fill_value=0)
    dist = {config.CLASSES[i]: {"count": int(counts[i]),
                                "share": round(float(counts[i]) / max(len(ds), 1), 3)} for i in LABELS}
    warnings_ = []
    if bad_dates:
        warnings_.append(f"{bad_dates} rows with unparseable date were dropped")
    if missing_values:
        warnings_.append("missing values found; rows missing label inputs cannot be used: " + json.dumps(missing_values))
    if dup_ids:
        warnings_.append(f"{dup_ids} duplicate dates inside the batch (kept the last record)")
    if overlap:
        warnings_.append(f"{overlap} batch dates already exist in the history (batch version replaces them)")
    absent = [k for k, v in dist.items() if v["count"] == 0]
    if absent:
        warnings_.append(f"class(es) absent from batch: {absent}; macro metrics average over classes that occur")
    if len(ds) < max(20, config.GATE_DAYS + 5):
        raise ValueError(f"only {len(ds)} usable labelled rows in batch; need at least "
                         f"{max(20, config.GATE_DAYS + 5)} (GATE_DAYS={config.GATE_DAYS} are held out)")

    report = {
        "checked_at": _now(),
        "required_columns_present": True,
        "n_rows_in_file": int(len(batch) + dup_ids + bad_dates),
        "n_rows_after_cleaning": int(len(batch)),
        "n_labelled_rows_usable": int(len(ds)),
        "date_range": [str(batch["date"].min().date()), str(batch["date"].max().date())],
        "missing_values_per_column": missing_values,
        "duplicate_ids_in_batch": dup_ids,
        "overlap_with_existing_history": overlap,
        "target_distribution": dist,
        "warnings": warnings_,
        "status": "passed_with_warnings" if warnings_ else "passed",
    }
    return batch, report


# ---------------------------------------------------------------- Task B
def classification_metrics(model, X: pd.DataFrame, y: pd.Series) -> dict:
    pred = model.predict(X)
    present = sorted(set(y) | set(pred))   # macro-average only over classes that occur (same as src/train.py)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", labels=present, zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", labels=present, zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", labels=present, zero_division=0)),
        "per_class": {config.CLASSES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                          "f1": float(f[i]), "support": int(s[i])} for i in LABELS},
        "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
    }


def evaluate_champion(champion, batch_ds: pd.DataFrame, gate_ds: pd.DataFrame, reports_dir) -> dict:
    report = {
        "evaluated_at": _now(),
        "model": "models/champion/model.joblib",
        "full_batch": classification_metrics(champion, batch_ds[config.FEATURES], batch_ds["target"]),
        "gate_window": classification_metrics(champion, gate_ds[config.FEATURES], gate_ds["target"]),
        "notes": "full_batch = every labelled day of the incoming batch (monitoring view). "
                 "gate_window = newest days only; used for the fair champion-vs-candidate comparison.",
    }
    _write_json(reports_dir / "evaluation_report.json", report)
    fb = report["full_batch"]
    md = ["# Champion evaluation on the new labelled batch", "",
          f"Rows evaluated: {fb['n']}", "",
          "| Metric | Full batch | Gate window |", "|---|---|---|"]
    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        md.append(f"| {k} | {fb[k]:.3f} | {report['gate_window'][k]:.3f} |")
    md += ["", "Per-class (full batch):", "", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for c, v in fb["per_class"].items():
        md.append(f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |")
    (reports_dir / "evaluation_report.md").write_text("\n".join(md) + "\n")
    return report


# ---------------------------------------------------------------- Task C
def retrain_candidate(existing_daily: pd.DataFrame, batch: pd.DataFrame, candidate_dir, reports_dir,
                      gate_days=config.GATE_DAYS):
    """Existing training data + new labelled batch (minus the gate window) -> candidate model."""
    combined = pd.concat([existing_daily, batch.assign(date=batch["date"].dt.strftime("%Y-%m-%d"))],
                         ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date").drop_duplicates("date", keep="last")
    ds = build_dataset(combined)                       # existing preprocessing pipeline
    if len(ds) < config.MIN_ROWS:
        raise SystemExit(f"Only {len(ds)} usable rows (need {config.MIN_ROWS}).")

    gate_start = ds.index.max() - pd.Timedelta(days=gate_days - 1)
    train, gate = ds[ds.index < gate_start], ds[ds.index >= gate_start]
    X_tr, y_tr = train[config.FEATURES], train["target"]

    cv = TimeSeriesSplit(n_splits=3)
    cv_scores = {n: float(cross_val_score(m, X_tr, y_tr, cv=cv, scoring="f1_macro").mean())
                 for n, m in candidate_models().items()}
    best = max(cv_scores, key=cv_scores.get)
    model = candidate_models()[best].fit(X_tr, y_tr)

    candidate_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, candidate_dir / "model.joblib")
    X_tr.tail(1500).to_csv(candidate_dir / "reference_features.csv", index_label="date")
    info = {
        "run_at": _now(), "selected_model": best, "cv_f1_macro": cv_scores,
        "n_rows": int(len(ds)), "n_train": int(len(train)), "n_holdout": int(len(gate)),
        "train_end": str(train.index.max().date()), "holdout_end": str(gate.index.max().date()),
        "features": config.FEATURES, "classes": config.CLASSES, "city": config.CITY,
        "status": "candidate (not promoted)",
    }
    return model, ds, train, gate, info


# ---------------------------------------------------------------- Task D
def quality_gate(champ: dict | None, cand: dict, min_improvement=config.MIN_IMPROVEMENT,
                 max_high_recall_drop=config.MAX_HIGH_RECALL_DROP) -> dict:
    """Eligible for promotion only if BOTH criteria hold on the same unseen gate window:
       1. candidate macro-F1 beats champion macro-F1 by more than `min_improvement`
       2. candidate recall on the 'High' class is not worse than champion's by > `max_high_recall_drop`
    """
    if champ is None:
        c1, c2 = True, True
        reasons = ["no usable champion exists -> candidate is eligible by default"]
    else:
        c1 = should_promote(cand["f1_macro"], champ["f1_macro"], min_improvement)
        high_c, high_k = cand["per_class"]["High"]["recall"], champ["per_class"]["High"]["recall"]
        c2 = high_c >= high_k - max_high_recall_drop
        reasons = [
            f"criterion 1 (macro-F1 {cand['f1_macro']:.3f} vs champion {champ['f1_macro']:.3f}, "
            f"required margin > {min_improvement}): {'PASS' if c1 else 'FAIL'}",
            f"criterion 2 (High-class recall {high_c:.3f} vs champion {high_k:.3f}, "
            f"max allowed drop {max_high_recall_drop}): {'PASS' if c2 else 'FAIL'}",
        ]
    eligible = bool(c1 and c2)
    return {
        "decision": "ELIGIBLE_FOR_PROMOTION" if eligible else "RETAIN_CHAMPION",
        "eligible_for_promotion": eligible,
        "primary_metric": "f1_macro",
        "metric_justification": (
            "Macro-F1 is the primary metric because the risk classes are imbalanced (see "
            "target_distribution) and accuracy would let a model ignore the rare 'High' class; macro-F1 "
            "weights Low/Moderate/High equally and combines precision and recall. It is also the metric "
            "the production daily-retrain gate (src/train.py) already uses, so decisions are consistent. "
            "A secondary guard on High-class recall prevents promoting a model that misses more "
            "high-risk days, the costliest error for this application."),
        "criteria_passed": {"macro_f1_improvement": bool(c1), "high_recall_guard": bool(c2)},
        "reasons": reasons,
        "champion_metrics": champ, "candidate_metrics": cand,
    }


def log_to_mlflow(info: dict, gate: dict, candidate_dir, reports_dir) -> str:
    """Track the retraining run in MLflow if it is installed (skipped silently otherwise)."""
    try:
        import mlflow
    except ImportError:
        print("[monitor] mlflow not installed; skipping experiment tracking")
        return "skipped (mlflow not installed)"
    import os
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"file:{config.ROOT / 'mlruns'}"))
    mlflow.set_experiment("respiratory-risk-monitoring")
    with mlflow.start_run(run_name=f"retrain-{info['run_at']}"):
        mlflow.log_params({"selected_model": info["selected_model"], "n_train": info["n_train"],
                           "n_gate": info["n_holdout"], "train_end": info["train_end"]})
        cand = gate["candidate_metrics"]
        mlflow.log_metrics({"candidate_f1": cand["f1_macro"], "candidate_accuracy": cand["accuracy"],
                            "candidate_high_recall": cand["per_class"]["High"]["recall"],
                            **({"champion_f1": gate["champion_metrics"]["f1_macro"]}
                               if gate["champion_metrics"] else {})})
        mlflow.set_tag("decision", gate["decision"])
        mlflow.log_artifact(str(candidate_dir / "model.joblib"))
        for f in ("evaluation_report.json", "quality_gate.json", "data_validation.json"):
            if (reports_dir / f).exists():
                mlflow.log_artifact(str(reports_dir / f))
    return "logged"


def run(data_path=config.DATA_PATH, batch_path=config.BATCH_PATH, model_dir=config.MODEL_DIR,
        candidate_dir=config.CANDIDATE_DIR, reports_dir=config.REPORTS_DIR,
        gate_days=config.GATE_DAYS, use_mlflow=True) -> dict:
    # ---- Task A
    daily = pd.read_csv(data_path)
    raw_batch = pd.read_csv(batch_path)
    batch_start = pd.to_datetime(raw_batch["date"]).min()
    existing = daily[pd.to_datetime(daily["date"]) < batch_start]   # history the champion era knew about
    batch, val = validate_batch(raw_batch, existing)
    _write_json(reports_dir / "data_validation.json", val)
    print(f"[A] batch validated: {val['status']}; {val['n_rows_after_cleaning']} rows "
          f"{val['date_range'][0]}..{val['date_range'][1]}; target distribution "
          + ", ".join(f"{k}={v['count']} ({v['share']:.0%})" for k, v in val["target_distribution"].items()))
    for w in val["warnings"]:
        print(f"    warning: {w}")

    # ---- Task C (built first so that the gate window is defined once, B and D then use it)
    model, ds, train, gate_ds, info = retrain_candidate(existing, batch, candidate_dir, reports_dir, gate_days)
    batch_ds = ds[(ds.index >= batch["date"].min()) & (ds.index <= batch["date"].max())]

    # ---- Task B
    champion_path = model_dir / "model.joblib"
    champion = joblib.load(champion_path) if champion_path.exists() else None
    champ_eval, champ_gate = None, None
    if champion is not None:
        try:
            champ_eval = evaluate_champion(champion, batch_ds, gate_ds, reports_dir)
            champ_gate = champ_eval["gate_window"]
            fb = champ_eval["full_batch"]
            print(f"[B] champion on new batch (n={fb['n']}): acc={fb['accuracy']:.3f} "
                  f"P={fb['precision_macro']:.3f} R={fb['recall_macro']:.3f} F1={fb['f1_macro']:.3f}")
        except Exception as e:
            print(f"[B] champion could not be evaluated ({e}); treating as absent")
    else:
        print("[B] no champion found; skipping evaluation")

    cand_gate = classification_metrics(model, gate_ds[config.FEATURES], gate_ds["target"])
    print(f"[C] candidate '{info['selected_model']}' trained on {info['n_train']} rows "
          f"(through {info['train_end']}) -> {candidate_dir / 'model.joblib'}")

    # ---- Task D
    gate = quality_gate(champ_gate, cand_gate)
    gate.update({"decided_at": _now(), "gate_window": [str(gate_ds.index.min().date()), str(gate_ds.index.max().date())],
                 "n_gate_rows": int(len(gate_ds)), "candidate_dir": str(candidate_dir),
                 "champion_overwritten": False})
    gate["mlflow"] = log_to_mlflow(info, gate, candidate_dir, reports_dir) if use_mlflow else "disabled"
    _write_json(reports_dir / "quality_gate.json", gate)
    info["gate_decision"] = gate["decision"]
    info["challenger"] = {"f1_macro": cand_gate["f1_macro"], "accuracy": cand_gate["accuracy"]}  # keys the API reads
    _write_json(candidate_dir / "metadata.json", info)
    with open(reports_dir / "decision_log.jsonl", "a") as f:
        f.write(json.dumps({"at": gate["decided_at"], "decision": gate["decision"], "action": "none (champion untouched)",
                            "champion_f1": champ_gate["f1_macro"] if champ_gate else None,
                            "candidate_f1": cand_gate["f1_macro"], "reasons": gate["reasons"]}) + "\n")
    print(f"[D] {gate['decision']}")
    for r in gate["reasons"]:
        print(f"    {r}")
    print("    champion NOT modified." + (" Run `python -m src.monitor promote` to promote it." if gate["eligible_for_promotion"] else ""))
    return {"validation": val, "evaluation": champ_eval, "candidate": info, "gate": gate}


# ---------------------------------------------------------------- explicit, logged promotion
def promote(model_dir=config.MODEL_DIR, candidate_dir=config.CANDIDATE_DIR, reports_dir=config.REPORTS_DIR) -> bool:
    gate_path = reports_dir / "quality_gate.json"
    if not gate_path.exists() or not (candidate_dir / "model.joblib").exists():
        print("[promote] no gate decision / candidate found - run `python -m src.monitor` first")
        return False
    gate = json.loads(gate_path.read_text())
    if not gate.get("eligible_for_promotion"):
        print("[promote] candidate is NOT eligible; champion retained")
        return False
    if model_dir.exists() and (model_dir / "model.joblib").exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive = model_dir.parent / "archive" / f"champion_{stamp}"
        shutil.copytree(model_dir, archive)                    # keep the old champion for rollback
    else:
        archive = None
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model.joblib", "reference_features.csv", "metadata.json"):
        shutil.copy2(candidate_dir / name, model_dir / name)
    gate.update({"champion_overwritten": True, "promoted_at": _now(),
                 "previous_champion_archived_to": str(archive) if archive else None})
    _write_json(gate_path, gate)
    with open(reports_dir / "decision_log.jsonl", "a") as f:
        f.write(json.dumps({"at": gate["promoted_at"], "decision": "PROMOTED", "action": "candidate copied to champion",
                            "archived_to": gate["previous_champion_archived_to"]}) + "\n")
    print(f"[promote] candidate promoted to champion (old one archived at {archive})")
    return True


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="run", choices=["run", "make-batch", "promote"])
    ap.add_argument("--days", type=int, default=config.BATCH_DAYS, help="rows for make-batch")
    args = ap.parse_args(argv)
    try:
        if args.command == "make-batch":
            make_batch(days=args.days)
        elif args.command == "promote":
            promote()
        else:
            run()
    except ValueError as e:
        sys.exit(f"[monitor] validation failed: {e}")


if __name__ == "__main__":
    main()
