"""Planner -> operator file formats.

Today that is one thing: the pathway CSV v2 of `docs/ARIS2_CONTRACTS.md` §1,
the same file the deployed GUI produces and `rtff_pathway_exec.py` loads, plus
the seven optional joint columns.  See `docs/EXPORT_PATHWAY.md`.

    python -m aris_sixarm.export.pathway --schedule out/X_schedule.npz \
        --program out/X_program.json --rig proposed --tool lateral --out out/pathways/

`from aris_sixarm.export import export` works, but the import is LAZY: an eager
`from .pathway import ...` here would put the submodule in `sys.modules` before
`python -m aris_sixarm.export.pathway` executes it, which runpy warns about.
"""
_LAZY = ("ArmPathway", "CSV_COLUMNS", "EXPORT_VERSION", "FK_TOL_DEG",
         "FK_TOL_M", "LIFT_M", "Stroke", "base_transform", "build_arm_pathway",
         "export", "paper_z_base", "quat_R", "quat_angle_deg", "quat_xyzw",
         "quats_from_R", "strokes_of", "tool_offsets", "verify_rows",
         "write_pathway")

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        from . import pathway
        return getattr(pathway, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
