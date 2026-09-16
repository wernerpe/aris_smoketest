# Vendored: `franka_analytical_ik`

The analytic IK C++ bindings the planner calls on every lattice step. This is
the **only** dependency of `aris_sixarm` that is not on PyPI, and it used to be
the only thing a fresh clone could not install. It is vendored here so that a
clone of this repository is self-contained: nothing to fetch, nothing to carry
from a site bundle, no bazel.

| | |
|---|---|
| upstream | `git@github.com:wernerpe/franka_analytical_ik.git` (`github.com/wernerpe/franka_analytical_ik`) |
| commit | **`0d38d9667b7c1d45acd65e698911e76437405533`** (`0d38d96`) |
| commit title | *Batch entry points: cross the pybind boundary once per lattice, not 6k times* |
| license | Apache-2.0 — `LICENSE`, byte-identical to upstream's |
| solver | He et al., analytic FR3/Panda IK (`paper_preprint.pdf` upstream, not vendored) |

## Install

```bash
pip install ./third_party/franka_analytical_ik
```

Needs `pybind11` (in `requirements-site.txt`, and in this directory's
`pyproject.toml` build requirements) and Eigen headers at `/usr/include/eigen3`
(`sudo apt install libeigen3-dev`; `EIGEN_INCLUDE=/path/to/eigen` overrides).
It compiles in about five seconds.

Verify, including that the fast path is live:

```bash
python -c "import aris_sixarm.ik as ik; print(ik._IK.__file__); print('batch:', ik.has_batch())"
```

Expect a path inside your venv's `site-packages/franka_analytical_ik/` and
`batch: True`.

## Why this commit, and not a later one

`0d38d96` is the commit that added the **batch** entry points — `solve_batch`,
`fk_batch`, `tip_jacobian_batch`. `aris_sixarm/ik.py` detects them at call time
and runs a python loop without them: an older commit is *correct* but roughly
an order of magnitude slower per lattice. Do not float this pin. If you bump
it, re-run the equivalence check below and say so here.

## Why a setuptools shim and not the upstream build

Upstream builds with bazel (`build_wheel.sh` → `bazel build //:franka_ik_wheel`).
Two things make that the wrong route here: the wheel target hardcodes
`python_tag = "cp310"` and `MODULE.bazel` pins a 3.10 toolchain, while the
planner runs python 3.12; and a toolchain fetch needs network and builds a very
large cache (102 GB on the workstation that has one).

`setup.py` here is a setuptools + pybind11 shim over the **identical** sources.
It was checked, not assumed: the shim-built extension exports the same six
entry points as the bazel-built `.so` this project has planned with since
August, and over **300 random poses × 4 branches × 7 joints** the two agree to
**max absolute difference 0.0**, with identical NaN masks. It is the same
solver, compiled by a different front end.

## What is here, and what was left behind

Vendored — everything the extension needs to build and import:

```
LICENSE                                       Apache-2.0, as upstream
pyproject.toml                                build backend (not upstream's; upstream has none)
setup.py                                      the shim (not upstream's)
franka_analytical_ik/__init__.py              upstream, unmodified
franka_analytical_ik/_franka_ik.pyi           upstream, unmodified
franka_analytical_ik/franka_analytic_ik_bindings.cpp   upstream, unmodified
franka_analytical_ik/franka_ik_He.hpp         upstream, unmodified
```

The four files under `franka_analytical_ik/` are byte-identical to
`git show 0d38d96:<path>` upstream. **Do not edit them**; a fix belongs
upstream, followed by a re-pin here.

Deliberately not vendored: `BUILD`, `MODULE.bazel`, `MODULE.bazel.lock`,
`.bazeliskrc`, `build_wheel.sh` (the bazel build this shim replaces),
`paper_preprint.pdf` (766 kB), `README.md`, `QUICKSTART.md`,
`BUILD_INSTRUCTIONS.md`, `test/`. All of them are one `git clone` away
upstream, and none of them is needed to build or run.

## A note on which entry points `aris_sixarm` uses

`aris_sixarm/ik.py` calls the **raw** `_franka_ik.solve_ik` / `solve_ik_cc`
(hand-TCP pose, column-major flat 16), *not* the higher-level `SolveIK` /
`SolveIKCC` wrappers in `__init__.py`, which take the flange pose and add the
0.1034 m hand offset themselves. Do not mix the two. The module docstring of
`aris_sixarm/ik.py` is the authority on this.
