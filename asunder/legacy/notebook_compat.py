"""Legacy notebook-compatible aliases."""

from __future__ import annotations

import warnings

from asunder.orchestrator import run_csd_decomposition


def CSD_decomposition(*args, **kwargs):
    """Backward-compatible alias for :func:`asunder.run_csd_decomposition`."""
    warnings.warn(
        "`asunder.legacy.notebook_compat.CSD_decomposition` is deprecated. "
        "Use `asunder.run_csd_decomposition`.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_csd_decomposition(*args, **kwargs)
