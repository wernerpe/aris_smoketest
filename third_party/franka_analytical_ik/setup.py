"""Portable build of the wernerpe/franka_analytical_ik pybind11 extension.

Replaces the upstream bazel build.  The bazel wheel target is pinned to
`python_tag = "cp310"` (upstream's root BUILD file) and `MODULE.bazel` pins a
3.10 toolchain, while the planner runs python 3.12; a toolchain fetch also
needs network and builds a very large cache.  Same two source files, same six
entry points; setuptools + pybind11 build them for whatever python runs pip,
in about five seconds, with no bazel and no network.

    pip install ./third_party/franka_analytical_ik

Set EIGEN_INCLUDE if Eigen is not at /usr/include/eigen3
(Ubuntu/Debian: `sudo apt install libeigen3-dev`).

See VENDOR.md for the upstream URL, the pinned commit, and the equivalence
check against the bazel-built extension.
"""
import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

EIGEN = os.environ.get("EIGEN_INCLUDE", "/usr/include/eigen3")

setup(
    name="franka_analytical_ik",
    version="1.0.0",
    packages=["franka_analytical_ik"],
    package_data={"franka_analytical_ik": ["*.pyi"]},
    ext_modules=[
        Pybind11Extension(
            "franka_analytical_ik._franka_ik",
            ["franka_analytical_ik/franka_analytic_ik_bindings.cpp"],
            include_dirs=[EIGEN, "franka_analytical_ik"],
            cxx_std=17,
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
