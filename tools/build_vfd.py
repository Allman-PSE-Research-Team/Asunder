"""Build and stage the optional standalone C++17 VFD library for this platform."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universal2", action="store_true", help="Build for Intel and ARM on macOS 11+")
    args = parser.parse_args()
    if args.universal2 and sys.platform != "darwin":
        parser.error("--universal2 requires a macOS build host")
    cmake = shutil.which("cmake")
    if cmake is None:
        parser.error("Install CMake and a C++17 compiler, then place cmake on PATH.")
    root = Path(__file__).resolve().parents[1]
    source = root / "native" / "vfd"
    arch = "universal2" if args.universal2 else platform.machine()
    build = root / "build" / "vfd" / f"{sys.platform}-{arch}"
    stage = root / "asunder" / "load_balancing" / "algorithms" / "_native"

    configure = [cmake, "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"]
    if args.universal2:
        configure += ["-DCMAKE_OSX_ARCHITECTURES=x86_64;arm64", "-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0"]
    subprocess.run(configure, check=True)
    subprocess.run([cmake, "--build", str(build), "--config", "Release", "--parallel"], check=True)
    subprocess.run([cmake, "--install", str(build), "--config", "Release", "--prefix", str(stage)], check=True)
    shutil.copyfile(source / "LICENSE", stage / "VFD-LICENSE")


if __name__ == "__main__":
    main()
