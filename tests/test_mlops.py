"""MLOps patterns: tracker fallback, identity pins, contract tests, MLflow replay."""

import sys

import pytest
import torch
import torch.nn as nn

from cosmo_gas_fields.mlops import (
    IdentityMismatch,
    Tracker,
    assert_identity,
    assert_step_contract,
    file_sha256,
    overfit_one_batch,
)


def test_tracker_falls_back_to_csv_without_mlflow(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)  # `import mlflow` raises ImportError
    with Tracker("Stage0-Test", {"model_type": "t"}, csv_dir=tmp_path) as trk:
        assert not trk.active
        trk.log_params({"lr": 1e-3})
        for s in range(3):
            trk.log_metric(s, "loss", 1.0 / (s + 1))
    rows = (tmp_path / "Stage0-Test_metrics.csv").read_text().splitlines()
    assert rows[0] == "step,metric,value" and len(rows) == 4
    assert rows[1].startswith("0,loss,1")


def test_identity_pin_match_and_mismatch(tmp_path):
    p = tmp_path / "cube.bin"
    p.write_bytes(b"\x00\x01\x02" * 1000)
    log = []
    assert_identity(p, "sha256", file_sha256(p), log)
    assert log[-1]["verdict"] == "MATCH"
    with pytest.raises(IdentityMismatch):
        assert_identity(p, "md5", "0" * 32, log)
    assert log[-1]["verdict"] == "MISMATCH"
    with pytest.raises(ValueError):
        assert_identity(p, "crc", "x")


def test_overfit_one_batch_passes_on_learnable_target():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))
    x = torch.randn(8, 4)
    y = x[:, :1] * 2.0 + 1.0
    rec = overfit_one_batch(model, (x, y), nn.functional.mse_loss, steps=200, lr=1e-2)
    assert rec["pass"] and rec["verdict"] == "PASS"
    assert len(rec["losses"]) == 201


def test_overfit_one_batch_fails_on_frozen_model():
    model = nn.Linear(2, 1)
    for p in model.parameters():
        p.requires_grad_(False)
    model.weight.requires_grad_(True)
    with torch.no_grad():
        model.weight.zero_()
    x = torch.zeros(4, 2)  # zero input: weight gradient is zero, loss cannot move
    y = torch.ones(4, 1)
    rec = overfit_one_batch(model, (x, y), nn.functional.mse_loss, steps=20)
    assert not rec["pass"]


def test_step_contract():
    losses = [1.0] + [0.5] * 99
    assert_step_contract(50, losses, pred_std=0.0)          # not the contract step: no-op
    assert_step_contract(100, losses, pred_std=0.1)         # passes
    with pytest.raises(AssertionError):
        assert_step_contract(100, losses, pred_std=0.0)      # collapsed prediction
    with pytest.raises(AssertionError):
        assert_step_contract(100, [0.5] * 100, pred_std=0.1)  # loss did not drop


def test_mlflow_replay_preserves_step_history(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")  # newer MLflow gates file stores
    from mlflow.tracking import MlflowClient

    from cosmo_gas_fields.mlops.mlflow_replay import replay_store

    src = tmp_path / "src_store"
    dst = tmp_path / "dst_store"
    src_client = MlflowClient(tracking_uri=src.as_uri())
    exp_id = src_client.create_experiment("offline/worker")
    run = src_client.create_run(exp_id, run_name="Stage1-Offline", tags={"stage": "1"})
    src_client.log_param(run.info.run_id, "lr", "0.001")
    for s in range(5):
        src_client.log_metric(run.info.run_id, "loss", 1.0 / (s + 1), step=s)
    src_client.set_terminated(run.info.run_id)

    mapping = replay_store(src, dst.as_uri(), "host/track", extra_tags={"compute": "hpc"})
    assert len(mapping) == 1
    dst_client = MlflowClient(tracking_uri=dst.as_uri())
    dst_run_id = list(mapping.values())[0]
    hist = dst_client.get_metric_history(dst_run_id, "loss")
    assert [m.step for m in sorted(hist, key=lambda m: m.step)] == [0, 1, 2, 3, 4]
    dst_run = dst_client.get_run(dst_run_id)
    assert dst_run.data.params["lr"] == "0.001"
    assert dst_run.data.tags["stage"] == "1" and dst_run.data.tags["compute"] == "hpc"
    assert dst_run.data.tags["source_run_id"] == run.info.run_id
