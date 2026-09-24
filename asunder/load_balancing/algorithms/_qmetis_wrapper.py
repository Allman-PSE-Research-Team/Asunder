"""Internal ctypes binding for Asunder's bundled QMETIS library.

This module is adapted from ``metis.py`` by Ken Watford, but deliberately
uses QMETIS's option layout and loads only the library path supplied by
Asunder. It is not a general binding for stock METIS.

The original wrapper is distributed under the MIT License:

Copyright (c) 2012 Ken Watford

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import annotations

import ctypes
import math
import operator
import os
from collections.abc import Iterable, Sequence
from typing import Any

QMETIS_NOPTIONS = 40
QMETIS_MODULARITY_SCALE = 1_000_000
QMETIS_RESOLUTION_SCALE = 1_000_000

METIS_OK = 1
METIS_ERROR_INPUT = -2
METIS_ERROR_MEMORY = -3
METIS_ERROR = -4

METIS_OPTION_PTYPE = 0
METIS_OPTION_OBJTYPE = 1
METIS_OPTION_CTYPE = 2
METIS_OPTION_IPTYPE = 3
METIS_OPTION_RTYPE = 4
METIS_OPTION_DBGLVL = 5
METIS_OPTION_NIPARTS = 6
METIS_OPTION_NITER = 7
METIS_OPTION_NCUTS = 8
METIS_OPTION_SEED = 9
METIS_OPTION_ONDISK = 10
METIS_OPTION_MINCONN = 11
METIS_OPTION_CONTIG = 12
METIS_OPTION_COMPRESS = 13
METIS_OPTION_CCORDER = 14
METIS_OPTION_PFACTOR = 15
METIS_OPTION_NSEPS = 16
METIS_OPTION_UFACTOR = 17
METIS_OPTION_NUMBERING = 18
METIS_OPTION_DROPEDGES = 19
METIS_OPTION_NO2HOP = 20
METIS_OPTION_TWOHOP = 21
METIS_OPTION_FAST = 22
METIS_OPTION_HELP = 23
METIS_OPTION_TPWGTS = 24
METIS_OPTION_NCOMMON = 25
METIS_OPTION_NOOUTPUT = 26
METIS_OPTION_BALANCE = 27
METIS_OPTION_GTYPE = 28
METIS_OPTION_UBVEC = 29
METIS_OPTION_MODRESOLUTION = 30

METIS_OBJTYPE_MOD = 3

_OPTION_INDICES = {
    "ptype": METIS_OPTION_PTYPE,
    "objtype": METIS_OPTION_OBJTYPE,
    "ctype": METIS_OPTION_CTYPE,
    "iptype": METIS_OPTION_IPTYPE,
    "rtype": METIS_OPTION_RTYPE,
    "dbglvl": METIS_OPTION_DBGLVL,
    "niparts": METIS_OPTION_NIPARTS,
    "niter": METIS_OPTION_NITER,
    "ncuts": METIS_OPTION_NCUTS,
    "seed": METIS_OPTION_SEED,
    "ondisk": METIS_OPTION_ONDISK,
    "minconn": METIS_OPTION_MINCONN,
    "contig": METIS_OPTION_CONTIG,
    "compress": METIS_OPTION_COMPRESS,
    "ccorder": METIS_OPTION_CCORDER,
    "pfactor": METIS_OPTION_PFACTOR,
    "nseps": METIS_OPTION_NSEPS,
    "ufactor": METIS_OPTION_UFACTOR,
    "numbering": METIS_OPTION_NUMBERING,
    "dropedges": METIS_OPTION_DROPEDGES,
    "no2hop": METIS_OPTION_NO2HOP,
    "twohop": METIS_OPTION_TWOHOP,
    "fast": METIS_OPTION_FAST,
    "help": METIS_OPTION_HELP,
    "tpwgts": METIS_OPTION_TPWGTS,
    "ncommon": METIS_OPTION_NCOMMON,
    "nooutput": METIS_OPTION_NOOUTPUT,
    "balance": METIS_OPTION_BALANCE,
    "gtype": METIS_OPTION_GTYPE,
    "ubvec": METIS_OPTION_UBVEC,
    "modresolution": METIS_OPTION_MODRESOLUTION,
}

_ENUM_VALUES = {
    "ptype": {"rb": 0, "kway": 1},
    "objtype": {"cut": 0, "vol": 1, "node": 2, "mod": METIS_OBJTYPE_MOD},
    "ctype": {"rm": 0, "shem": 1},
    "iptype": {"grow": 0, "random": 1, "edge": 2, "node": 3},
    "rtype": {"fm": 0, "greedy": 1, "sep2sided": 2, "sep1sided": 3},
    "gtype": {"dual": 0, "nodal": 1},
}


class QMETISError(RuntimeError):
    """Base exception raised by the internal QMETIS binding."""


class QMETISInputError(QMETISError):
    """Indicate that QMETIS rejected its input."""


class QMETISMemoryError(QMETISError):
    """Indicate that QMETIS could not allocate required memory."""


def _integer(value: Any, description: str) -> int:
    """Return an integer-compatible value or raise a descriptive error."""

    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{description} must be an integer.") from error


def _idx_value(value: Any, idx_t: type, description: str) -> int:
    """Return an integer after checking the selected QMETIS ``idx_t`` range."""

    integer = _integer(value, description)
    bits = ctypes.sizeof(idx_t) * 8
    minimum = -(2 ** (bits - 1))
    maximum = 2 ** (bits - 1) - 1
    if integer < minimum or integer > maximum:
        raise OverflowError(f"{description} exceeds the QMETIS idx_t range.")
    return integer


def _normalise_option_name(name: str) -> str:
    """Map a user option name to the compact QMETIS spelling."""

    normalised = name.lower()
    for prefix in ("metis_option_", "option_"):
        if normalised.startswith(prefix):
            normalised = normalised[len(prefix) :]
    return normalised


def _flatten_real_values(
    values: Iterable[Any],
    *,
    expected: int,
    description: str,
) -> list[float]:
    """Flatten a one- or two-dimensional sequence of real values."""

    flattened: list[float] = []
    for value in values:
        try:
            flattened.append(float(value))
        except (TypeError, ValueError):
            if isinstance(value, (str, bytes)):
                raise
            flattened.extend(float(item) for item in value)
    if len(flattened) != expected:
        raise ValueError(
            f"{description} must contain {expected} values; got {len(flattened)}."
        )
    if not all(math.isfinite(value) for value in flattened):
        raise ValueError(f"{description} must contain only finite values.")
    return flattened


class QMETISBinding:
    """Bind one explicitly selected QMETIS shared library.

    Parameters
    ----------
    library_path : os.PathLike or str
        Absolute path to the QMETIS shared library shipped in the wheel.
    idx_width : {32, 64}
        ``IDXTYPEWIDTH`` used to compile that library.
    real_width : {32, 64}
        ``REALTYPEWIDTH`` used to compile that library.
    """

    def __init__(
        self,
        library_path: os.PathLike[str] | str,
        *,
        idx_width: int,
        real_width: int,
    ) -> None:
        path = os.path.abspath(os.fspath(library_path))
        if idx_width not in (32, 64):
            raise ValueError("idx_width must be 32 or 64.")
        if real_width not in (32, 64):
            raise ValueError("real_width must be 32 or 64.")

        self.library_path = path
        self.idx_t = ctypes.c_int32 if idx_width == 32 else ctypes.c_int64
        self.real_t = ctypes.c_float if real_width == 32 else ctypes.c_double
        self._library = ctypes.CDLL(path)
        self._configure_functions()

    def _configure_functions(self) -> None:
        """Assign the exact public QMETIS C signatures."""

        idx_pointer = ctypes.POINTER(self.idx_t)
        real_pointer = ctypes.POINTER(self.real_t)

        self._set_default_options = self._library.METIS_SetDefaultOptions
        self._set_default_options.argtypes = [idx_pointer]
        self._set_default_options.restype = ctypes.c_int

        signature = [
            idx_pointer,  # nvtxs
            idx_pointer,  # ncon
            idx_pointer,  # xadj
            idx_pointer,  # adjncy
            idx_pointer,  # vwgt
            idx_pointer,  # vsize
            idx_pointer,  # adjwgt
            idx_pointer,  # nparts
            real_pointer,  # tpwgts
            real_pointer,  # ubvec
            idx_pointer,  # options
            idx_pointer,  # objval
            idx_pointer,  # part
        ]
        self._part_graph_kway = self._library.METIS_PartGraphKway
        self._part_graph_kway.argtypes = signature
        self._part_graph_kway.restype = ctypes.c_int

        self._part_graph_recursive = self._library.METIS_PartGraphRecursive
        self._part_graph_recursive.argtypes = signature
        self._part_graph_recursive.restype = ctypes.c_int

    @staticmethod
    def _check_status(status: int, operation: str) -> None:
        """Raise the matching Python exception for a QMETIS status."""

        if status == METIS_OK:
            return
        if status == METIS_ERROR_INPUT:
            raise QMETISInputError(f"QMETIS rejected the input to {operation}.")
        if status == METIS_ERROR_MEMORY:
            raise QMETISMemoryError(
                f"QMETIS ran out of memory during {operation}."
            )
        raise QMETISError(
            f"QMETIS failed during {operation} with status {status}."
        )

    def _make_options(
        self,
        *,
        resolution: float,
        supplied: dict[str, Any],
    ) -> ctypes.Array:
        """Build a QMETIS option array with explicit modularity settings."""

        resolution = float(resolution)
        if not math.isfinite(resolution) or resolution < 0:
            raise ValueError("resolution must be a finite nonnegative value.")

        options = (self.idx_t * QMETIS_NOPTIONS)()
        self._check_status(
            self._set_default_options(options),
            "METIS_SetDefaultOptions",
        )

        for original_name, value in supplied.items():
            name = _normalise_option_name(original_name)
            if name not in _OPTION_INDICES:
                raise TypeError(f"Unknown QMETIS option {original_name!r}.")
            if name == "modresolution":
                raise TypeError(
                    "Pass resolution as the dedicated Python argument, not "
                    "as a raw modresolution option."
                )

            if isinstance(value, str):
                choices = _ENUM_VALUES.get(name, {})
                try:
                    value = choices[value.lower()]
                except KeyError as error:
                    raise ValueError(
                        f"Invalid value {value!r} for QMETIS option {name!r}."
                    ) from error
            options[_OPTION_INDICES[name]] = _idx_value(
                value,
                self.idx_t,
                f"QMETIS option {name!r}",
            )

        requested_objective = int(options[METIS_OPTION_OBJTYPE])
        if "objtype" in {_normalise_option_name(key) for key in supplied}:
            if requested_objective != METIS_OBJTYPE_MOD:
                raise ValueError(
                    "Asunder's QMETIS binding supports only the modularity "
                    "objective."
                )

        fixed_resolution = round(resolution * QMETIS_RESOLUTION_SCALE)
        maximum = 2 ** (ctypes.sizeof(self.idx_t) * 8 - 1) - 1
        if fixed_resolution > maximum:
            raise OverflowError("The scaled QMETIS resolution exceeds idx_t.")

        options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_MOD
        options[METIS_OPTION_MODRESOLUTION] = fixed_resolution
        return options

    def _make_graph(
        self,
        adjacency: Sequence[Sequence[Any]],
        nodew: Sequence[Any] | None,
        nodesz: Sequence[Any] | None,
    ) -> tuple[Any, ...]:
        """Convert adjacency and optional vertex data into QMETIS arrays."""

        vertex_count = len(adjacency)
        xadj_values = [0]
        adjncy_values: list[int] = []
        adjwgt_values: list[int] = []
        weighted = False

        for row, entries in enumerate(adjacency):
            for entry in entries:
                try:
                    weighted_entry = (
                        not isinstance(entry, (str, bytes))
                        and len(entry) == 2
                    )
                except TypeError:
                    weighted_entry = False
                if weighted_entry:
                    neighbour, weight = entry
                    weighted = True
                else:
                    neighbour, weight = entry, 1
                neighbour = _idx_value(
                    neighbour,
                    self.idx_t,
                    "Adjacency index",
                )
                weight = _idx_value(
                    weight,
                    self.idx_t,
                    "Adjacency weight",
                )
                if neighbour < 0 or neighbour >= vertex_count:
                    raise ValueError(
                        f"Adjacency row {row} contains out-of-range vertex "
                        f"{neighbour}."
                    )
                if weight < 0:
                    raise ValueError("QMETIS adjacency weights must be nonnegative.")
                adjncy_values.append(neighbour)
                adjwgt_values.append(weight)
            xadj_values.append(len(adjncy_values))

        idx_t = self.idx_t
        xadj = (idx_t * len(xadj_values))(*xadj_values)
        adjncy = (idx_t * len(adjncy_values))(*adjncy_values)
        adjwgt = (
            (idx_t * len(adjwgt_values))(*adjwgt_values) if weighted else None
        )

        ncon = 1
        vwgt = None
        if nodew is not None:
            if len(nodew) != vertex_count:
                raise ValueError("nodew must contain one entry per vertex.")
            rows: list[list[int]] = []
            for value in nodew:
                try:
                    row_values = [
                        _idx_value(value, self.idx_t, "Vertex weight")
                    ]
                except TypeError:
                    row_values = [
                        _idx_value(item, self.idx_t, "Vertex weight")
                        for item in value
                    ]
                if not row_values:
                    raise ValueError("Vertex weight rows cannot be empty.")
                if any(item < 0 for item in row_values):
                    raise ValueError("QMETIS vertex weights must be nonnegative.")
                rows.append(row_values)
            ncon = len(rows[0])
            if any(len(row) != ncon for row in rows):
                raise ValueError(
                    "Every vertex must provide the same number of weights."
                )
            flattened = [item for row in rows for item in row]
            vwgt = (idx_t * len(flattened))(*flattened)

        vsize = None
        if nodesz is not None:
            if len(nodesz) != vertex_count:
                raise ValueError("nodesz must contain one entry per vertex.")
            sizes = [
                _idx_value(value, self.idx_t, "Vertex size")
                for value in nodesz
            ]
            if any(value < 0 for value in sizes):
                raise ValueError("QMETIS vertex sizes must be nonnegative.")
            vsize = (idx_t * len(sizes))(*sizes)

        return xadj, adjncy, adjwgt, vwgt, vsize, ncon

    def part_graph(
        self,
        adjacency: Sequence[Sequence[Any]],
        *,
        nparts: int = 2,
        resolution: float = 1.0,
        tpwgts: Sequence[Any] | None = None,
        ubvec: Sequence[Any] | None = None,
        recursive: bool = False,
        nodew: Sequence[Any] | None = None,
        nodesz: Sequence[Any] | None = None,
        **options: Any,
    ) -> tuple[float, list[int]]:
        """Partition an adjacency list with QMETIS's modularity objective.

        Parameters
        ----------
        adjacency : sequence of sequences
            Each entry is a neighbor index or ``(neighbor, integer_weight)``.
        nparts : int, default=2
            Number of requested parts.
        resolution : float, default=1.0
            Modularity resolution :math:`\\gamma`. It is encoded using
            QMETIS's fixed scale of one million.
        tpwgts, ubvec, recursive, nodew, nodesz
            Standard METIS partitioning inputs retained by the adapted binding.
        **options
            Additional QMETIS options by name, such as ``seed``, ``niter``,
            ``ncuts``, and ``contig``.

        Returns
        -------
        modularity : float
            Native modularity objective converted from QMETIS's fixed scale.
        parts : list[int]
            Part identifier for each adjacency row.
        """

        vertex_count = len(adjacency)
        nparts = _idx_value(nparts, self.idx_t, "nparts")
        if vertex_count == 0:
            raise ValueError("adjacency must contain at least one vertex.")
        if nparts < 2 or nparts > vertex_count:
            raise ValueError("nparts must be between 2 and the vertex count.")

        xadj, adjncy, adjwgt, vwgt, vsize, ncon_value = self._make_graph(
            adjacency,
            nodew,
            nodesz,
        )
        options_array = self._make_options(
            resolution=resolution,
            supplied=options,
        )

        real_t = self.real_t
        target_weights = None
        if tpwgts is not None:
            values = _flatten_real_values(
                tpwgts,
                expected=nparts * ncon_value,
                description="tpwgts",
            )
            target_weights = (real_t * len(values))(*values)

        imbalance = None
        if ubvec is not None:
            values = _flatten_real_values(
                ubvec,
                expected=ncon_value,
                description="ubvec",
            )
            if any(value < 1.0 for value in values):
                raise ValueError("Every ubvec value must be at least 1.")
            imbalance = (real_t * len(values))(*values)

        idx_t = self.idx_t
        nvtxs = idx_t(vertex_count)
        ncon = idx_t(ncon_value)
        part_count = idx_t(nparts)
        objective = idx_t()
        parts = (idx_t * vertex_count)()
        partition = (
            self._part_graph_recursive if recursive else self._part_graph_kway
        )
        status = partition(
            ctypes.byref(nvtxs),
            ctypes.byref(ncon),
            xadj,
            adjncy,
            vwgt,
            vsize,
            adjwgt,
            ctypes.byref(part_count),
            target_weights,
            imbalance,
            options_array,
            ctypes.byref(objective),
            parts,
        )
        self._check_status(
            status,
            getattr(partition, "__name__", "graph partitioning"),
        )
        return (
            objective.value / QMETIS_MODULARITY_SCALE,
            [int(value) for value in parts],
        )
