"""Setuptools command customization for data-only native platform wheels."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
from setuptools.command.build_py import build_py

PROJECT_ROOT = Path(__file__).resolve().parent
NATIVE_DIRECTORY = (
    PROJECT_ROOT
    / "asunder"
    / "load_balancing"
    / "algorithms"
    / "_native"
)
NATIVE_PACKAGE = "asunder.load_balancing.algorithms._native"
PLATFORM_LIBRARIES = {
    "win32": {"qmetis.dll", "asunder_vfd.dll"},
    "darwin": {"libqmetis.dylib", "libasunder_vfd.dylib"},
    "linux": {"libqmetis.so", "libasunder_vfd.so"},
}
HOST_LIBRARIES = PLATFORM_LIBRARIES.get(sys.platform, set())


def _contains_native_library() -> bool:
    """Return whether a QMETIS or VFD runtime has been staged."""

    return any((NATIVE_DIRECTORY / name).is_file() for name in HOST_LIBRARIES)


class NativeBuildPy(build_py):
    """Package only host libraries when Windows and WSL share a checkout."""

    def find_data_files(self, package, src_dir):
        files = super().find_data_files(package, src_dir)
        if package != NATIVE_PACKAGE:
            return files
        foreign_libraries = set().union(*PLATFORM_LIBRARIES.values()) - HOST_LIBRARIES
        return [path for path in files if Path(path).name not in foreign_libraries]


class PlatformWheel(_bdist_wheel):
    """Tag wheels with bundled ctypes libraries as Python-ABI independent."""

    def finalize_options(self) -> None:
        """Mark a wheel non-pure when it contains a staged native runtime."""

        super().finalize_options()
        self.root_is_pure = not _contains_native_library()

    def get_tag(self) -> tuple[str, str, str]:
        """Return a Python-independent ABI tag for a native platform wheel."""

        python_tag, abi_tag, platform_tag = super().get_tag()
        if self.root_is_pure:
            return python_tag, abi_tag, platform_tag

        override = os.environ.get("ASUNDER_WHEEL_PLATFORM_TAG")
        return "py3", "none", override or platform_tag


class NativeAwareDistribution(Distribution):
    """Route staged native builds through the wheel ``platlib`` scheme."""

    def has_ext_modules(self) -> bool:
        """Route a staged native runtime through setuptools' platlib scheme."""

        return _contains_native_library()


setup(
    cmdclass={"bdist_wheel": PlatformWheel, "build_py": NativeBuildPy},
    distclass=NativeAwareDistribution,
)
