"""
Type stubs for franka_analytical_ik

Analytical Inverse Kinematics solver for Franka Emika Panda robot.
"""

from typing import List
import numpy as np
import numpy.typing as npt

def solve_ik(
    O_T_EE_array: npt.NDArray[np.float64],
    q7: float,
    q_actual_array: npt.NDArray[np.float64],
) -> List[npt.NDArray[np.float64]]:
    """
    Compute inverse kinematics for Franka Emika Panda robot.

    This function returns up to 4 different joint configurations for a given 
    end-effector pose. Solutions that violate joint limits are set to NaN.

    Parameters
    ----------
    O_T_EE_array : numpy.ndarray
        Desired Cartesian pose as a 4x4 transformation matrix in column-major 
        format (flattened to 16 elements). Must be float64 dtype.
    q7 : float
        Last joint angle (wrist) as redundant parameter in radians.
    q_actual_array : numpy.ndarray
        Current joint configuration (7 elements) in radians. Must be float64 dtype.

    Returns
    -------
    list of numpy.ndarray
        List of 4 solutions, each as a 7-element float64 array. Invalid solutions 
        contain NaN values.

    Raises
    ------
    ValueError
        If O_T_EE_array doesn't have 16 elements or q_actual_array doesn't have 7 elements.

    Examples
    --------
    >>> import numpy as np
    >>> from franka_analytical_ik import solve_ik
    >>> O_T_EE = np.eye(4, dtype=np.float64).flatten('F')  # Column-major format
    >>> q7 = 0.0
    >>> q_actual = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0], dtype=np.float64)
    >>> solutions = solve_ik(O_T_EE, q7, q_actual)
    >>> # Check which solutions are valid
    >>> valid_solutions = [sol for sol in solutions if not np.any(np.isnan(sol))]
    
    Notes
    -----
    - The transformation matrix must be in column-major (Fortran) order
    - Joint limits are checked; solutions outside limits return NaN
    - Up to 4 solutions may be returned representing different configurations 
      (e.g., elbow-up, elbow-down, etc.)
    - Assumes Franka Hand is installed (d7e = 0.2104m)
    """
    ...

def solve_ik_cc(
    O_T_EE_array: npt.NDArray[np.float64],
    q7: float,
    q_actual_array: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute case-consistent inverse kinematics for Franka Emika Panda robot.

    This function returns only the solution that belongs to the same "case" as 
    the current joint configuration. This prevents unexpected switching between 
    different solution cases (e.g., elbow-up and elbow-down) during continuous 
    motion planning.

    Parameters
    ----------
    O_T_EE_array : numpy.ndarray
        Desired Cartesian pose as a 4x4 transformation matrix in column-major 
        format (flattened to 16 elements). Must be float64 dtype.
    q7 : float
        Last joint angle (wrist) as redundant parameter in radians.
    q_actual_array : numpy.ndarray
        Current joint configuration (7 elements) in radians. Must be float64 dtype.

    Returns
    -------
    numpy.ndarray
        Single solution as a 7-element float64 array. If no valid solution exists in 
        the current case, returns NaN values.

    Raises
    ------
    ValueError
        If O_T_EE_array doesn't have 16 elements or q_actual_array doesn't have 7 elements.

    Examples
    --------
    >>> import numpy as np
    >>> from franka_analytical_ik import solve_ik_cc
    >>> O_T_EE = np.eye(4, dtype=np.float64).flatten('F')  # Column-major format
    >>> q7 = 0.0
    >>> q_actual = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0], dtype=np.float64)
    >>> solution = solve_ik_cc(O_T_EE, q7, q_actual)
    >>> if not np.any(np.isnan(solution)):
    ...     print("Valid solution found!")
    
    Notes
    -----
    - The transformation matrix must be in column-major (Fortran) order
    - Returns solution consistent with current joint configuration "case"
    - Useful for trajectory planning to avoid configuration jumps
    - Assumes Franka Hand is installed (d7e = 0.2104m)
    """
    ...


def solve_batch(
    O_T_EE_flat: npt.NDArray[np.float64],
    q7: npt.NDArray[np.float64],
    q_actual: npt.NDArray[np.float64],
    pos_tol: float = 1e-9,
    rot_tol: float = 1e-9,
) -> npt.NDArray[np.float64]:
    """
    Batched analytic IK with in-C++ FK verification.

    Runs `franka_IK_EE` for every row and certifies each of the four returned
    branches against the pose that was asked for.  The solver CLAMPS at the
    workspace boundary instead of failing: it can return a finite, in-limits
    configuration for a pose it misses by centimetres, and nothing downstream
    can tell that apart from a real solution.  Genuine solutions reproduce the
    pose to ~1e-12, so the 1e-9 default separates them unambiguously.

    No joint-limit filtering beyond the solver's own (PANDA limits) is applied;
    an FR3 caller must re-filter.  The GIL is released for the batch loop.

    Parameters
    ----------
    O_T_EE_flat : (N,16) float64
        Hand-TCP poses, each a 4x4 flattened COLUMN-major.
    q7 : (N,) float64
        Redundancy parameter (joint 7) per row.
    q_actual : (N,7) or (7,) float64
        Seed configuration; a single (7,) seed is broadcast over the batch.
    pos_tol : float
        Position tolerance of the FK verification, metres.
    rot_tol : float
        Frobenius tolerance on the rotation block.

    Returns
    -------
    (N,4,7) float64
        Branch solutions.  A branch that was NaN, or that failed the FK
        verification, is an all-NaN row.
    """
    ...


def fk_batch(
    q: npt.NDArray[np.float64],
    tcp: float = 0.2104,
) -> dict:
    """
    Batched forward kinematics (Craig modified DH, then Rz(-pi/4) and Tz(tcp)).

    Parameters
    ----------
    q : (N,7) float64
    tcp : float
        Hand-TCP offset along tool z after the flange twist.

    Returns
    -------
    dict
        "T"   (N,4,4) float64 hand-TCP poses in link0
        "pts" (N,9,3) float64 chain points: base origin, J1..J7 origins, TCP
    """
    ...


def tip_jacobian_batch(
    q: npt.NDArray[np.float64],
    pen_ext: float,
    tcp: float = 0.2104,
) -> npt.NDArray[np.float64]:
    """
    Batched ANALYTIC geometric position Jacobian of the pen tip.

    tip = hand-TCP + pen_ext along tool z; column i is z_i x (p_tip - p_i) with
    z_i, p_i the axis and origin of joint i from the modified-DH chain.  Exact,
    and a drop-in replacement for a 14-FK central finite difference.

    Parameters
    ----------
    q : (N,7) float64
    pen_ext : float
        Pen length beyond the hand TCP, metres.
    tcp : float
        Hand-TCP offset along tool z.

    Returns
    -------
    (N,3,7) float64
    """
    ...


has_batch: bool
"""Feature flag: present only on builds that carry the batch entry points."""
