from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).resolve().parent
SOURCES = [
    ROOT / "nanopokerlib" / "cpp" / "bindings.cpp",
    ROOT / "nanopokerlib" / "cpp" / "guandan" / "guandan.cpp",
]


class GuandanBuildExt(build_ext):
    """Small offline-friendly pybind builder for the repository toolchains.

    Windows CI for this project exposes MinGW rather than MSVC/CMake, so the
    command creates a local GNU import library from pythonXY.dll and invokes
    g++ directly. Linux/macOS use the host C++ compiler with a normal shared
    extension build.
    """

    def build_extension(self, ext: Extension) -> None:
        import pybind11

        output = Path(self.get_ext_fullpath(ext.name)).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)
        include_args = [
            f"-I{pybind11.get_include()}",
            f"-I{sysconfig.get_paths()['include']}",
            f"-I{ROOT / 'nanopokerlib' / 'cpp'}",
        ]
        sources = [str(path) for path in SOURCES]
        if platform.system() == "Windows":
            cxx = os.environ.get("CXX") or shutil.which("g++")
            gendef = shutil.which("gendef")
            dlltool = shutil.which("dlltool")
            if not cxx or not gendef or not dlltool:
                raise RuntimeError("Windows build requires g++, gendef, and dlltool")
            python_dll = Path(sys.base_prefix) / f"python{sys.version_info.major}{sys.version_info.minor}.dll"
            if not python_dll.is_file():
                raise RuntimeError(f"Python DLL not found: {python_dll}")
            def_file = build_temp / f"python{sys.version_info.major}{sys.version_info.minor}.def"
            import_lib = build_temp / f"libpython{sys.version_info.major}{sys.version_info.minor}.a"
            gendef_command = [gendef, str(python_dll)]
            print("Guandan build gendef command:", subprocess.list2cmdline(gendef_command))
            subprocess.check_call(gendef_command, cwd=build_temp)
            generated = build_temp / python_dll.with_suffix(".def").name
            if generated != def_file:
                generated.replace(def_file)
            dlltool_command = [dlltool, "-d", str(def_file), "-l", str(import_lib), "-D", python_dll.name]
            print("Guandan build dlltool command:", subprocess.list2cmdline(dlltool_command))
            subprocess.check_call(dlltool_command, cwd=build_temp)
            command = [
                cxx,
                "-O3",
                "-DNDEBUG",
                "-std=c++20",
                "-shared",
                "-static-libgcc",
                "-static-libstdc++",
                "-Wl,--enable-auto-import",
                *include_args,
                *sources,
                str(import_lib),
                "-Wl,-Bstatic",
                "-lwinpthread",
                "-Wl,-Bdynamic",
                "-o",
                str(output),
            ]
        else:
            cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
            if not cxx:
                raise RuntimeError("a C++20 compiler is required")
            command = [
                cxx,
                "-O3",
                "-DNDEBUG",
                "-std=c++20",
                "-shared",
                "-fPIC",
                *include_args,
                *sources,
                "-o",
                str(output),
            ]
            if platform.system() == "Darwin":
                command[1:1] = ["-undefined", "dynamic_lookup"]
        print("Guandan build Python executable:", sys.executable)
        print("Guandan build Python include:", sysconfig.get_paths()["include"])
        print("Guandan build pybind11 include:", pybind11.get_include())
        print("Guandan build compiler:", cxx)
        print("Guandan build command:", subprocess.list2cmdline(command))
        subprocess.check_call(command, cwd=ROOT)


setup(
    ext_modules=[Extension("nanopokerlib._guandan_cpp", sources=[])],
    cmdclass={"build_ext": GuandanBuildExt},
)
