"""Experiment tracker with an offline fallback and an unconditional CSV mirror.

Failure this prevents: a run that trains for hours on a cluster node with no
route to the tracking server, then exits 0 with no metrics anywhere. Every
metric is ALWAYS written to a local CSV; MLflow is used when reachable and
silently dropped mid-run if it stops responding.
"""

from __future__ import annotations

import csv
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional


class CSVMirror:
    """Append-only (step, metric, value) CSV, flushed on every write."""

    def __init__(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = open(path, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(["step", "metric", "value"])

    def log(self, step: int, metric: str, value: float) -> None:
        self._w.writerow([step, metric, f"{value:.10g}"])
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class Tracker:
    """MLflow wrapper: nullcontext fallback + CSV mirror.

    Parameters
    ----------
    run_name, tags
        Passed to ``mlflow.start_run``. Use a stage-prefixed run name and a
        fixed set of mandatory tags (model type, stage, dataset variant, seed,
        git commit) so runs stay filterable.
    csv_dir
        Where the mirror CSV goes (``<csv_dir>/<run_name>_metrics.csv``).
    experiment
        MLflow experiment name (hierarchical, e.g. ``"project/track"``).
    tracking_uri
        Defaults to ``$MLFLOW_TRACKING_URI`` or a local server.
    """

    def __init__(self, run_name: str, tags: Dict[str, str], csv_dir: Path,
                 experiment: str = "cosmo-gas-fields/default",
                 tracking_uri: Optional[str] = None) -> None:
        self.csv = CSVMirror(Path(csv_dir) / f"{run_name}_metrics.csv")
        self._mlflow = None
        self._ctx = nullcontext()
        self.active = False
        try:
            import mlflow

            mlflow.set_tracking_uri(
                tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
            )
            mlflow.set_experiment(experiment)
            self._ctx = mlflow.start_run(run_name=run_name, tags=tags)
            self._mlflow = mlflow
            self.active = True
        except Exception as exc:  # noqa: BLE001 — degrade to local CSV only
            print(f"[tracker] MLflow unavailable ({exc!r}); CSV mirror only.", flush=True)

    def __enter__(self):
        self._ctx.__enter__()
        return self

    def __exit__(self, *a):
        self.csv.close()
        return self._ctx.__exit__(*a)

    def log_metric(self, step: int, metric: str, value: float) -> None:
        self.csv.log(step, metric, value)
        if self._mlflow is not None:
            try:
                self._mlflow.log_metric(metric, value, step=step)
            except Exception:  # noqa: BLE001
                self._mlflow = None
                self.active = False
                print("[tracker] MLflow dropped mid-run; CSV mirror only.", flush=True)

    def log_params(self, params: Dict) -> None:
        if self._mlflow is not None:
            try:
                self._mlflow.log_params(params)
            except Exception:  # noqa: BLE001
                pass
