import json

import pandas as pd
import pytest

from src import monitor, train
from src.ingest import make_synthetic


def _setup(tmp_path, n=500, batch_days=80):
    data = tmp_path / "daily.csv"
    daily = make_synthetic(n)
    daily.to_csv(data, index=False)
    batch = tmp_path / "batch.csv"
    daily.tail(batch_days).to_csv(batch, index=False)
    dirs = dict(model_dir=tmp_path / "champ", candidate_dir=tmp_path / "cand", reports_dir=tmp_path / "rep")
    return data, batch, dirs


def test_validation_flags_problems_and_missing_columns():
    daily = make_synthetic(300)
    existing, batch = daily.head(200), daily.tail(100).copy()
    dirty = pd.concat([batch, batch.head(3)])             # duplicate ids
    dirty.loc[dirty.index[5], "pm10_mean"] = None          # missing value
    _, rep = monitor.validate_batch(dirty, existing)
    assert rep["duplicate_ids_in_batch"] == 3
    assert rep["missing_values_per_column"] == {"pm10_mean": 1}
    assert set(rep["target_distribution"]) == {"Low", "Moderate", "High"}
    with pytest.raises(ValueError):
        monitor.validate_batch(batch.drop(columns=["pm2_5_mean"]), existing)


def test_quality_gate_rules():
    def m(f1, high):
        return {"f1_macro": f1, "per_class": {"High": {"recall": high}}}
    assert monitor.quality_gate(None, m(0.5, 0.5))["eligible_for_promotion"]
    assert monitor.quality_gate(m(0.70, 0.8), m(0.75, 0.8))["eligible_for_promotion"]
    assert not monitor.quality_gate(m(0.70, 0.8), m(0.70, 0.8))["eligible_for_promotion"]   # tie keeps champion
    assert not monitor.quality_gate(m(0.70, 0.8), m(0.60, 0.9))["eligible_for_promotion"]   # worse F1
    assert not monitor.quality_gate(m(0.70, 0.9), m(0.80, 0.7))["eligible_for_promotion"]   # misses High days


def test_full_workflow_never_overwrites_champion_and_logs_decision(tmp_path):
    data, batch, d = _setup(tmp_path)
    train.run(data_path=data, model_dir=d["model_dir"], reports_dir=d["reports_dir"])                         # create a champion
    before = (d["model_dir"] / "model.joblib").read_bytes()
    out = monitor.run(data_path=data, batch_path=batch, use_mlflow=False, **d)
    assert (d["model_dir"] / "model.joblib").read_bytes() == before        # untouched
    assert (d["candidate_dir"] / "model.joblib").exists()
    for f in ("data_validation.json", "evaluation_report.json", "quality_gate.json", "decision_log.jsonl"):
        assert (d["reports_dir"] / f).exists()
    gate = json.loads((d["reports_dir"] / "quality_gate.json").read_text())
    assert gate["decision"] in {"ELIGIBLE_FOR_PROMOTION", "RETAIN_CHAMPION"}
    assert gate["champion_overwritten"] is False
    assert {"accuracy", "precision_macro", "recall_macro", "f1_macro"} <= set(out["evaluation"]["full_batch"])


def test_promote_only_when_eligible(tmp_path):
    data, batch, d = _setup(tmp_path)
    train.run(data_path=data, model_dir=d["model_dir"], reports_dir=d["reports_dir"])
    monitor.run(data_path=data, batch_path=batch, use_mlflow=False, **d)
    gp = d["reports_dir"] / "quality_gate.json"
    gate = json.loads(gp.read_text())
    gate["eligible_for_promotion"] = False
    gp.write_text(json.dumps(gate))
    assert monitor.promote(**d) is False
    gate["eligible_for_promotion"] = True
    gp.write_text(json.dumps(gate))
    assert monitor.promote(**d) is True
    assert list((d["model_dir"].parent / "archive").iterdir())             # old champion archived
    assert json.loads(gp.read_text())["champion_overwritten"] is True
