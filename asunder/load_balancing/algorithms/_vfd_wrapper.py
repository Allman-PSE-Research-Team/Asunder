"""ctypes interface to the platform wheel's C++ load-balancing search."""

from __future__ import annotations

import ctypes as ct
import operator
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


class _Options(ct.Structure):
    _fields_ = [
        ("groups", ct.c_int32), ("lower", ct.c_int64), ("upper", ct.c_int64),
        ("restarts", ct.c_int32), ("local_iters", ct.c_int32),
        ("tabu_max_steps", ct.c_int32), ("shake_rounds", ct.c_int32),
        ("w_coassoc", ct.c_double), ("allow_block_splitting", ct.c_int32),
        ("check_invariants", ct.c_int32), ("seed", ct.c_uint64),
    ]


_COUNTERS = (
    "restarts_attempted", "feasible_restarts", "moves", "swaps", "ejections", "splits",
    "shake_moves", "tabu_steps", "candidate_moves", "candidate_swaps", "candidate_ejections",
)


class _Stats(ct.Structure):
    _fields_ = [
        ("found", ct.c_int32), ("objective_B_sum", ct.c_double), ("native_seconds", ct.c_double),
        *((name, ct.c_uint64) for name in _COUNTERS),
    ]


def _library_path() -> Path:
    if sys.platform == "win32":
        filename = "asunder_vfd.dll"
    elif sys.platform == "darwin":
        filename = "libasunder_vfd.dylib"
    else:
        filename = "libasunder_vfd.so"
    return Path(__file__).resolve().parent / "_native" / filename


@lru_cache(maxsize=1)
def load_library() -> ct.CDLL:
    """Bind the packaged VFD library on its first requested use."""
    path = _library_path()
    try:
        library = ct.CDLL(str(path))
        run_search = library.asunder_vfd_search_v1
        action_delta = library.asunder_vfd_action_delta_v1
    except (OSError, AttributeError) as exc:
        raise RuntimeError(
            f"Could not load the C++ VFD library at {path}. Install a supported Asunder "
            "platform wheel or run 'python tools/build_vfd.py' in a source checkout."
        ) from exc

    matrix = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=("F_CONTIGUOUS", "ALIGNED"))
    integers = np.ctypeslib.ndpointer(dtype=np.int32, flags=("C_CONTIGUOUS", "ALIGNED"))
    loads = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
    changes = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
    error = ct.POINTER(ct.c_char)
    run_search.argtypes = [
        ct.c_int32, matrix, matrix, loads, ct.c_size_t, integers, ct.c_int32, integers,
        integers, loads, ct.POINTER(_Options), loads, ct.POINTER(_Stats), error, ct.c_size_t,
    ]
    action_delta.argtypes = [
        ct.c_int32, matrix, matrix, integers, ct.c_int32, integers, integers, ct.c_int32,
        ct.c_int32, ct.c_int32, ct.c_double, changes, error, ct.c_size_t,
    ]
    run_search.restype = action_delta.restype = ct.c_int32
    return library


def _matrix_size(B, C):
    for value in (B, C):
        if (not isinstance(value, np.ndarray) or value.dtype != np.float64
                or value.ndim != 2 or value.shape[0] != value.shape[1]
                or not value.flags.f_contiguous or not value.flags.aligned):
            raise ValueError("Component matrices must be square, aligned, Fortran-contiguous float64 arrays.")
    if B.shape != C.shape or not 0 < B.shape[0] <= np.iinfo(np.int32).max - 63:
        raise ValueError("Invalid component matrix dimensions.")
    return B.shape[0]


def _integer_array(values, dtype):
    array = np.asarray(values)
    if array.size:
        limits = np.iinfo(dtype)
        if (array.dtype.kind not in "iu" or int(array.min()) < limits.min
                or int(array.max()) > limits.max):
            raise ValueError(f"Expected an array of {np.dtype(dtype)} integers.")
    return np.require(array, dtype=dtype, requirements=("C", "A"))


def _integer(value, ctype):
    value = operator.index(value)
    if ctype(value).value != value:
        raise ValueError(f"Integer outside the supported native range: {value}.")
    return value


def _pack_blocks(blocks, n):
    flat = []
    offsets = [0]
    for block in blocks:
        flat.extend(block)
        offsets.append(len(flat))
    if len(flat) != n or len(offsets) - 1 > n:
        raise ValueError("Blocks must partition the component IDs exactly once.")
    components = _integer_array(flat, np.int32)
    if components.shape != (n,):
        raise ValueError("Invalid component ID dimensions.")
    return np.asarray(offsets, dtype=np.int32), components


def _call(function, *args):
    message = ct.create_string_buffer(512)
    status = function(*args, message, len(message))
    if status:
        exception = {1: ValueError, 2: MemoryError}.get(status, RuntimeError)
        raise exception(message.value.decode("utf-8", errors="replace"))


def search(W_B, C_comp, weights, cannot_pairs, blocks, target, groups, lower, upper,
           restarts, local_iters, tabu_max_steps, shake_rounds, w_coassoc, allow_block_splitting,
           check_invariants, seed):
    """Search one community count with prepared component matrices and load targets."""
    n = _matrix_size(W_B, C_comp)
    weights = _integer_array(weights, np.int64)
    target = _integer_array(target, np.int64)
    groups = _integer(groups, ct.c_int32)
    if weights.shape != (n,) or target.shape != (groups,):
        raise ValueError("Component weights and target loads have inconsistent dimensions.")
    pairs = _integer_array(cannot_pairs, np.int32)
    if pairs.shape == (0,):
        pairs = pairs.reshape(0, 2)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("Each cannot-link must contain two component IDs.")
    offsets, components = _pack_blocks(blocks, n)
    options = _Options(
        groups, _integer(lower, ct.c_int64), _integer(upper, ct.c_int64),
        _integer(restarts, ct.c_int32), _integer(local_iters, ct.c_int32),
        _integer(tabu_max_steps, ct.c_int32), _integer(shake_rounds, ct.c_int32),
        w_coassoc, bool(allow_block_splitting), bool(check_invariants), _integer(seed, ct.c_uint64),
    )
    labels = np.empty(n, dtype=np.int64)
    stats = _Stats()
    _call(load_library().asunder_vfd_search_v1, n, W_B, C_comp, weights, len(pairs), pairs,
          len(offsets) - 1, offsets, components, target, ct.byref(options), labels, ct.byref(stats))
    return {
        "found": bool(stats.found),
        "component_labels": labels if stats.found else labels[:0],
        "objective_B_sum": stats.objective_B_sum,
        "native_seconds": stats.native_seconds,
        **{name: getattr(stats, name) for name in _COUNTERS},
    }


def _action_delta(W_B, C_comp, labels, blocks, first, second, destination, w_coassoc=0.05):
    """Compare incremental action gains to a full objective recomputation for tests."""
    n = _matrix_size(W_B, C_comp)
    labels = _integer_array(labels, np.int32)
    if labels.shape != (n,):
        raise ValueError("Diagnostic labels must contain one entry per component.")
    offsets, components = _pack_blocks(blocks, n)
    changes = np.empty(4, dtype=np.float64)
    _call(load_library().asunder_vfd_action_delta_v1, n, W_B, C_comp, labels,
          len(offsets) - 1, offsets, components, _integer(first, ct.c_int32),
          _integer(second, ct.c_int32), _integer(destination, ct.c_int32), w_coassoc, changes)
    return changes.tolist()
