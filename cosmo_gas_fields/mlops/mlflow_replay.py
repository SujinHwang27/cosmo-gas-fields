"""Replay an MLflow file-store into a tracking server, preserving per-step history.

Failure this prevents: training jobs on the HPC cluster (or any offline
worker) have no network route to the tracking server, so they log to a local
``file://`` store. Without a replay step those metrics never reach the shared
tracker, and per-step curves are lost if only final values are copied.
:func:`replay_run` recreates each source run in the destination (params,
tags, full metric history with original timestamps and steps, artifacts, and
terminal status); :func:`replay_store` walks every experiment/run in a store.

Requires ``mlflow`` (optional dependency). Recent MLflow versions refuse to
open a ``file://`` store unless ``MLFLOW_ALLOW_FILE_STORE=true``; the source
here is always a file store, so :func:`replay_store` sets that variable if unset.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def ensure_experiment(client, name: str) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is None:
        return client.create_experiment(name)
    return exp.experiment_id


def replay_run(src_client, dst_client, src_run_id: str, dst_experiment_id: str,
               src_store_root: Path, latest_only: bool = False,
               extra_tags: Optional[Dict[str, str]] = None) -> str:
    """Recreate one source run in the destination tracker; returns the new run id.

    ``latest_only=True`` logs only the final value per metric (fast path for
    SQLite-backed destinations with very long runs).
    """
    src_run = src_client.get_run(src_run_id)
    src_data = src_run.data
    src_info = src_run.info
    run_name = src_data.tags.get("mlflow.runName", src_run_id)
    tags = {k: v for k, v in src_data.tags.items() if not k.startswith("mlflow.")}
    tags.update({"imported_from_file_store": "true", "source_run_id": src_run_id})
    if extra_tags:
        tags.update(extra_tags)
    dst_run = dst_client.create_run(experiment_id=dst_experiment_id,
                                    start_time=src_info.start_time,
                                    run_name=run_name, tags=tags)
    dst_run_id = dst_run.info.run_id

    for k, v in src_data.params.items():
        dst_client.log_param(dst_run_id, k, v)

    for metric_key in src_data.metrics:
        history = src_client.get_metric_history(src_run_id, metric_key)
        if not history:
            continue
        for m in (history[-1:] if latest_only else history):
            dst_client.log_metric(dst_run_id, m.key, m.value, timestamp=m.timestamp, step=m.step)

    # The artifact_uri in meta.yaml points at the path on the machine that
    # wrote the store; resolve artifacts from the on-disk layout instead.
    src_artifact_path = Path(src_store_root) / src_info.experiment_id / src_run_id / "artifacts"
    if src_artifact_path.exists():
        for fp in src_artifact_path.rglob("*"):
            if fp.is_file():
                rel = fp.relative_to(src_artifact_path).parent.as_posix()
                dst_client.log_artifact(dst_run_id, str(fp),
                                        artifact_path=None if rel in ("", ".") else rel)

    dst_client.set_terminated(dst_run_id, status=src_info.status, end_time=src_info.end_time)
    return dst_run_id


def replay_store(src_store_root: Path, dst_tracking_uri: str, dst_experiment: str,
                 latest_only: bool = False, extra_tags: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Replay every run in a file-store; returns {source_run_id: dest_run_id}."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    from mlflow.tracking import MlflowClient

    src_store_root = Path(src_store_root).resolve()
    src_client = MlflowClient(tracking_uri=src_store_root.as_uri())
    dst_client = MlflowClient(tracking_uri=dst_tracking_uri)
    dst_experiment_id = ensure_experiment(dst_client, dst_experiment)
    replayed: Dict[str, str] = {}
    for src_exp in src_client.search_experiments():
        for src_run in src_client.search_runs(experiment_ids=[src_exp.experiment_id], max_results=1000):
            replayed[src_run.info.run_id] = replay_run(
                src_client, dst_client, src_run.info.run_id, dst_experiment_id,
                src_store_root=src_store_root, latest_only=latest_only, extra_tags=extra_tags,
            )
    return replayed
