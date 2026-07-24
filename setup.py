"""Setuptools command customization for data-only native platform wheels."""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel

PROJECT_ROOT = Path(__file__).resolve().parent
NATIVE_DIRECTORY = (
    PROJECT_ROOT
    / "asunder"
    / "load_balancing"
    / "algorithms"
    / "_native"
)
QMETIS_LIBRARY_NAMES = {
    "qmetis.dll",
    "libqmetis.dylib",
    "libqmetis.so",
}


def _contains_native_library() -> bool:
    """Return whether a recognized QMETIS runtime has been staged."""

    return any(
        path.is_file() and path.name in QMETIS_LIBRARY_NAMES
        for path in NATIVE_DIRECTORY.glob("*")
    )


class PlatformWheel(_bdist_wheel):
    """Tag wheels with bundled ctypes libraries as Python-ABI independent."""

    def finalize_options(self) -> None:
        """Mark a wheel non-pure when it contains a staged QMETIS runtime."""

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
    cmdclass={"bdist_wheel": PlatformWheel},
    distclass=NativeAwareDistribution,
)
