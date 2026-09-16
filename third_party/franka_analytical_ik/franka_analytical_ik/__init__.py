"""
Franka Analytical IK - Analytical Inverse Kinematics solver for Franka Emika Panda

This package provides fast analytical inverse kinematics solutions for the 
Franka Emika Panda 7-DOF robot manipulator.
"""

try:
    from . import _franka_ik
except ImportError:
    # Fallback for direct imports during development
    import _franka_ik

import numpy as np

__version__ = _franka_ik.__version__
__all__ = ['solve_ik', 'solve_ik_cc']

def SolveIK(X_W_EE : np.ndarray, 
            q_redundancy: float, 
            q_current :np.ndarray):
    """
    Compute inverse kinematics for Franka Emika Panda robot.

    This function returns up to 4 different joint configurations for a given 
    end-effector pose. Solutions that violate joint limits are set to NaN. 

    Parameters
    ----------
    X_W_EE : numpy.ndarray
        Desired Cartesian pose of endeffector flange (EE) as a 4x4 transformation matrix.

    q_redundancy : float
        Redundancy parameter for the solutions [0, 2*pi].

    q_actual_array : numpy.ndarray
        Current joint configuration (7 elements) in radians. Must be float64 dtype.

    Returns
    -------
    list of numpy.ndarray
        List of 4 solutions, each as a 7-element float64 array. Invalid solutions 
        contain NaN values.
    """
    assert X_W_EE.shape[0] == X_W_EE.shape[1]
    assert X_W_EE.shape[0] == 4
    X_W_EE_internal = X_W_EE.copy()
    #remove the hand offset
    X_EE_offset = np.eye(4)
    X_EE_offset[2, 3] = 0.1034  # offset in EE frame
    X_W_EE_internal = X_W_EE @ X_EE_offset
    O_T_EE_flat = X_W_EE_internal.flatten('F')
    solutions = _franka_ik.solve_ik(O_T_EE_flat, q_redundancy, q_current)
    return solutions

def SolveIKCC(X_W_EE : np.ndarray, 
            q_redundancy: float, 
            q_current :np.ndarray):
    
    """
    Compute inverse kinematics for Franka Emika Panda robot.

    This function returns a joint configuration with the same global configuration (case) as the current configuration for a given 
    end-effector pose. 

    Parameters
    ----------
    X_W_EE : numpy.ndarray
        Desired Cartesian pose of endeffector flange (EE) as a 4x4 transformation matrix.

    q_redundancy : float
        Redundancy parameter for the solutions [0, 2*pi].

    q_actual_array : numpy.ndarray
        Current joint configuration (7 elements) in radians. Must be float64 dtype.

    Returns
    -------
    numpy.ndarray
        A 7-element float64 array. If no solution is found it returns NaN.
    """

    assert X_W_EE.shape[0] == X_W_EE.shape[1]
    assert X_W_EE.shape[0] == 4
    X_W_EE_internal = X_W_EE.copy()
    #remove the hand offset
    X_EE_offset = np.eye(4)
    X_EE_offset[2, 3] = 0.1034  # offset in EE frame
    X_W_EE_internal = X_W_EE @ X_EE_offset
    O_T_EE_flat = X_W_EE_internal.flatten('F')
    solution = _franka_ik.solve_ik_cc(O_T_EE_flat, q_redundancy, q_current)
    return solution


# solve_ik = _franka_ik.solve_ik
# solve_ik_cc = _franka_ik.solve_ik_cc