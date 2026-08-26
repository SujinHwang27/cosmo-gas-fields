"""Identity pins: assert an input artifact is byte-identical to the one on record.

Failure this prevents: a scoring script silently reading a regenerated,
re-smoothed, or differently-ordered cube that has the same file name as the
one whose numbers are already recorded. Every load-bearing input is pinned by
hash, checked at load, and a mismatch aborts loudly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional


class IdentityMismatch(SystemExit):
    """Raised (as a SystemExit so it cannot be swallowed by broad excepts) on a pin mismatch."""


def file_sha256(path: Path, first_mb_only: bool = False) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        if first_mb_only:
            h.update(fh.read(1 << 20))
        else:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
    return h.hexdigest()


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_identity(path: Path, kind: str, pinned: str,
                    log: Optional[List[Dict]] = None) -> str:
    """Hash ``path`` with ``kind`` in {'sha256', 'sha256_first_mb', 'md5'} and compare to ``pinned``.

    Appends a record to ``log`` if given; raises :class:`IdentityMismatch` on mismatch.
    """
    path = Path(path)
    if kind == "sha256":
        observed = file_sha256(path)
    elif kind == "sha256_first_mb":
        observed = file_sha256(path, first_mb_only=True)
    elif kind == "md5":
        observed = file_md5(path)
    else:
        raise ValueError(f"unknown hash kind {kind!r}")
    verdict = "MATCH" if observed == pinned else "MISMATCH"
    if log is not None:
        log.append({"file": path.name, "kind": kind, "pinned": pinned,
                    "observed": observed, "verdict": verdict})
    print(f"[identity] {path.name} {kind}: {verdict}", flush=True)
    if verdict == "MISMATCH":
        raise IdentityMismatch(f"IDENTITY MISMATCH: {path.name} ({kind}) "
                               f"pinned={pinned} observed={observed}")
    return observed
