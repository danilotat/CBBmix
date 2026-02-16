"""Build script for CBBmix C++ BAF extension.

Compiles _baf.cpp against htslib via pybind11. If htslib is not found,
the extension is silently skipped and the package installs as pure-Python.
"""

import subprocess
import sys
from pathlib import Path

from setuptools import setup

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    def get_htslib_config():
        """Detect htslib via pkg-config, Homebrew, or system fallback."""
        include_dirs = []
        library_dirs = []

        # Tier 1: pkg-config
        try:
            cflags = (
                subprocess.check_output(
                    ["pkg-config", "--cflags", "htslib"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
            libs = (
                subprocess.check_output(
                    ["pkg-config", "--libs-only-L", "htslib"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            for flag in cflags.split():
                if flag.startswith("-I"):
                    include_dirs.append(flag[2:])
            for flag in libs.split():
                if flag.startswith("-L"):
                    library_dirs.append(flag[2:])
            return include_dirs, library_dirs
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Tier 2: Homebrew (macOS)
        if sys.platform == "darwin":
            try:
                prefix = (
                    subprocess.check_output(
                        ["brew", "--prefix", "htslib"], stderr=subprocess.DEVNULL
                    )
                    .decode()
                    .strip()
                )
                p = Path(prefix)
                if (p / "include").is_dir():
                    include_dirs.append(str(p / "include"))
                if (p / "lib").is_dir():
                    library_dirs.append(str(p / "lib"))
                return include_dirs, library_dirs
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

        # Tier 3: system fallback (empty — relies on default search paths)
        return include_dirs, library_dirs

    include_dirs, library_dirs = get_htslib_config()

    ext_modules = [
        Pybind11Extension(
            "CBBmix._baf",
            ["src/CBBmix/_baf.cpp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=["hts"],
            cxx_std=17,
        ),
    ]

    setup(
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
    )

except Exception as e:
    print(
        f"WARNING: Could not build C++ BAF extension ({e}). "
        "Installing without BAF support.",
        file=sys.stderr,
    )
    setup()
