#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "franka_ik_He.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Batch entry points.
//
// The per-call pybind crossing dominates the analytic solver itself (2-5 us of
// C++ under ~50 us of python), so a planner that solves a whole lattice pays
// for the boundary, not the maths.  The three functions below take whole
// arrays, do the loop (and the FK verification that has to accompany it) with
// the GIL released, and hand back one contiguous result.
//
// FK CONVENTION.  `fk_chain` below is a literal transcription of the reference
// python model (aris_sixarm/frames.py:fk): Craig modified DH with
//   (alpha_{i-1}, a_{i-1}, d_i) = (0,0,.333) (-pi/2,0,0) (pi/2,0,.316)
//                                 (pi/2,.0825,0) (-pi/2,-.0825,.384)
//                                 (pi/2,0,0) (pi/2,.088,0)
// followed by the flange twist Rz(-pi/4) and the hand offset Tz(0.2104) — the
// same d7e the He solver assumes.  Frame {i} of the modified-DH chain has its
// z axis ON joint i's axis, so the chain gives the geometric Jacobian directly.
//
// WHY THE FK VERIFICATION IS NOT OPTIONAL.  franka_IK_EE CLAMPS at the
// workspace boundary rather than failing: near the inner boundary it returns a
// finite, in-limits configuration for a pose it misses by up to ~2 cm.  Nothing
// downstream can tell that apart from a real solution.  Genuine solutions
// reproduce the requested pose to ~1e-12, clamped ones miss by mm to cm, so a
// 1e-9 threshold separates them with six orders of magnitude to spare.
// ---------------------------------------------------------------------------
namespace {

// modified DH: (alpha_{i-1}, a_{i-1}, d_i), i = 1..7
const double DH_ALPHA[7] = {0.0, -M_PI / 2.0, M_PI / 2.0, M_PI / 2.0,
                            -M_PI / 2.0, M_PI / 2.0, M_PI / 2.0};
const double DH_A[7] = {0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088};
const double DH_D[7] = {0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0};
const double TCP_D_DEFAULT = 0.2104;   // flange twist offset = the solver's d7e

struct FkChain {
    Eigen::Matrix4d T;            // hand-TCP pose in link0
    Eigen::Vector3d p[9];         // base origin, J1..J7 origins, TCP
    Eigen::Vector3d z[7];         // joint axes (z of frame {i}), in link0
};

inline void fk_chain(const double* q, double tcp, FkChain& out) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    out.p[0] = T.block<3, 1>(0, 3);
    for (int i = 0; i < 7; ++i) {
        const double al = DH_ALPHA[i], a = DH_A[i], d = DH_D[i], th = q[i];
        const double ca = std::cos(al), sa = std::sin(al);
        const double ct = std::cos(th), st = std::sin(th);
        Eigen::Matrix4d M;
        M << ct,    -st,    0.0, a,
             st*ca, ct*ca, -sa,  -sa*d,
             st*sa, ct*sa,  ca,   ca*d,
             0.0,   0.0,    0.0,  1.0;
        T = T * M;
        out.p[i + 1] = T.block<3, 1>(0, 3);
        out.z[i] = T.block<3, 1>(0, 2);
    }
    const double c = std::cos(-M_PI / 4.0), s = std::sin(-M_PI / 4.0);
    Eigen::Matrix4d E;
    E << c,   -s,  0.0, 0.0,
         s,    c,  0.0, 0.0,
         0.0, 0.0, 1.0, tcp,
         0.0, 0.0, 0.0, 1.0;
    T = T * E;
    out.T = T;
    out.p[8] = T.block<3, 1>(0, 3);
}

// Contiguous float64 view of an array, with a copy only when one is needed.
inline py::array_t<double, py::array::c_style | py::array::forcecast>
as_c64(py::array a) {
    return py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(a);
}

}  // namespace

PYBIND11_MODULE(_franka_ik, m) {
    m.doc() = "Analytical Inverse Kinematics solver for Franka Emika Panda robot";

    m.def("solve_ik", 
        [](py::array_t<double> O_T_EE_array, double q7, py::array_t<double> q_actual_array) {
            // Validate input sizes
            if (O_T_EE_array.size() != 16) {
                throw std::invalid_argument("O_T_EE_array must have 16 elements");
            }
            if (q_actual_array.size() != 7) {
                throw std::invalid_argument("q_actual_array must have 7 elements");
            }

            // Convert numpy arrays to std::array
            std::array<double, 16> O_T_EE;
            std::array<double, 7> q_actual;
            
            auto O_T_EE_buf = O_T_EE_array.unchecked<1>();
            auto q_actual_buf = q_actual_array.unchecked<1>();
            
            for (size_t i = 0; i < 16; i++) {
                O_T_EE[i] = O_T_EE_buf(i);
            }
            for (size_t i = 0; i < 7; i++) {
                q_actual[i] = q_actual_buf(i);
            }

            // Call the IK solver
            auto result = franka_IK_EE(O_T_EE, q7, q_actual);

            // Convert result to list of numpy arrays
            py::list solutions;
            for (const auto& sol : result) {
                py::array_t<double> sol_array(7);
                auto sol_buf = sol_array.mutable_unchecked<1>();
                for (size_t i = 0; i < 7; i++) {
                    sol_buf(i) = sol[i];
                }
                solutions.append(sol_array);
            }

            return solutions;
        },
        py::arg("O_T_EE_array"),
        py::arg("q7"),
        py::arg("q_actual_array"),
        R"pbdoc(
            Compute inverse kinematics for Franka Emika Panda robot.

            This function returns up to 4 different joint configurations for a given 
            end-effector pose. Solutions that violate joint limits are set to NaN.

            Parameters
            ----------
            O_T_EE_array : numpy.ndarray
                Desired Cartesian pose as a 4x4 transformation matrix in column-major 
                format (flattened to 16 elements)
            q7 : float
                Last joint angle (wrist) as redundant parameter in radians
            q_actual_array : numpy.ndarray
                Current joint configuration (7 elements) in radians

            Returns
            -------
            list of numpy.ndarray
                List of 4 solutions, each as a 7-element array. Invalid solutions 
                contain NaN values.

            Examples
            --------
            >>> import numpy as np
            >>> O_T_EE = np.eye(4).flatten('F')  # Column-major format
            >>> q7 = 0.0
            >>> q_actual = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
            >>> solutions = solve_ik(O_T_EE, q7, q_actual)
        )pbdoc"
    );

    m.def("solve_ik_cc", 
        [](py::array_t<double> O_T_EE_array, double q7, py::array_t<double> q_actual_array) {
            // Validate input sizes
            if (O_T_EE_array.size() != 16) {
                throw std::invalid_argument("O_T_EE_array must have 16 elements");
            }
            if (q_actual_array.size() != 7) {
                throw std::invalid_argument("q_actual_array must have 7 elements");
            }

            // Convert numpy arrays to std::array
            std::array<double, 16> O_T_EE;
            std::array<double, 7> q_actual;
            
            auto O_T_EE_buf = O_T_EE_array.unchecked<1>();
            auto q_actual_buf = q_actual_array.unchecked<1>();
            
            for (size_t i = 0; i < 16; i++) {
                O_T_EE[i] = O_T_EE_buf(i);
            }
            for (size_t i = 0; i < 7; i++) {
                q_actual[i] = q_actual_buf(i);
            }

            // Call the case-consistent IK solver
            auto result = franka_IK_EE_CC(O_T_EE, q7, q_actual);

            // Convert result to numpy array
            py::array_t<double> sol_array(7);
            auto sol_buf = sol_array.mutable_unchecked<1>();
            for (size_t i = 0; i < 7; i++) {
                sol_buf(i) = result[i];
            }

            return sol_array;
        },
        py::arg("O_T_EE_array"),
        py::arg("q7"),
        py::arg("q_actual_array"),
        R"pbdoc(
            Compute case-consistent inverse kinematics for Franka Emika Panda robot.

            This function returns only the solution that belongs to the same "case" as 
            the current joint configuration. This prevents unexpected switching between 
            different solution cases (e.g., elbow-up and elbow-down) during continuous 
            motion planning.

            Parameters
            ----------
            O_T_EE_array : numpy.ndarray
                Desired Cartesian pose as a 4x4 transformation matrix in column-major 
                format (flattened to 16 elements)
            q7 : float
                Last joint angle (wrist) as redundant parameter in radians
            q_actual_array : numpy.ndarray
                Current joint configuration (7 elements) in radians

            Returns
            -------
            numpy.ndarray
                Single solution as a 7-element array. If no valid solution exists in 
                the current case, returns NaN values.

            Examples
            --------
            >>> import numpy as np
            >>> O_T_EE = np.eye(4).flatten('F')  # Column-major format
            >>> q7 = 0.0
            >>> q_actual = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
            >>> solution = solve_ik_cc(O_T_EE, q7, q_actual)
        )pbdoc"
    );

    // ---------------------------------------------------------------- batch
    m.def("solve_batch",
        [](py::array O_T_EE_flat_in, py::array q7_in, py::array q_actual_in,
           double pos_tol, double rot_tol) {
            auto poses = as_c64(O_T_EE_flat_in);
            auto q7s = as_c64(q7_in);
            auto qa = as_c64(q_actual_in);

            if (poses.ndim() != 2 || poses.shape(1) != 16) {
                throw std::invalid_argument("O_T_EE_flat must be (N,16)");
            }
            const py::ssize_t N = poses.shape(0);
            if (q7s.ndim() != 1 || q7s.shape(0) != N) {
                throw std::invalid_argument("q7 must be (N,)");
            }
            bool broadcast_seed;
            if (qa.ndim() == 1 && qa.shape(0) == 7) {
                broadcast_seed = true;
            } else if (qa.ndim() == 2 && qa.shape(0) == N && qa.shape(1) == 7) {
                broadcast_seed = false;
            } else {
                throw std::invalid_argument("q_actual must be (N,7) or (7,)");
            }

            py::array_t<double> out({N, py::ssize_t(4), py::ssize_t(7)});
            const double* pose_p = poses.data();
            const double* q7_p = q7s.data();
            const double* qa_p = qa.data();
            double* out_p = out.mutable_data();

            {
                py::gil_scoped_release release;
                std::array<double, 16> O_T_EE;
                std::array<double, 7> seed;
                FkChain chain;
                for (py::ssize_t i = 0; i < N; ++i) {
                    const double* row = pose_p + i * 16;
                    for (int k = 0; k < 16; ++k) O_T_EE[k] = row[k];
                    const double* sp = broadcast_seed ? qa_p : qa_p + i * 7;
                    for (int k = 0; k < 7; ++k) seed[k] = sp[k];

                    // requested pose, column-major flat -> R (cols) and p
                    Eigen::Matrix3d R_req;
                    for (int c = 0; c < 3; ++c)
                        for (int r = 0; r < 3; ++r) R_req(r, c) = row[c * 4 + r];
                    Eigen::Vector3d p_req(row[12], row[13], row[14]);

                    auto sols = franka_IK_EE(O_T_EE, q7_p[i], seed);
                    double* dst = out_p + i * 28;
                    for (int b = 0; b < 4; ++b) {
                        const std::array<double, 7>& q = sols[b];
                        bool finite = true;
                        for (int k = 0; k < 7; ++k)
                            if (!std::isfinite(q[k])) { finite = false; break; }
                        bool keep = false;
                        if (finite) {
                            fk_chain(q.data(), TCP_D_DEFAULT, chain);
                            const double dp =
                                (chain.T.block<3, 1>(0, 3) - p_req).norm();
                            const double dR =
                                (chain.T.block<3, 3>(0, 0) - R_req).norm();
                            keep = (dp <= pos_tol) && (dR <= rot_tol);
                        }
                        if (keep) {
                            for (int k = 0; k < 7; ++k) dst[b * 7 + k] = q[k];
                        } else {
                            for (int k = 0; k < 7; ++k)
                                dst[b * 7 + k] =
                                    std::numeric_limits<double>::quiet_NaN();
                        }
                    }
                }
            }
            return out;
        },
        py::arg("O_T_EE_flat"), py::arg("q7"), py::arg("q_actual"),
        py::arg("pos_tol") = 1e-9, py::arg("rot_tol") = 1e-9,
        R"pbdoc(
            Batched analytic IK with in-C++ FK verification.

            Runs `franka_IK_EE` for every row and certifies each of the four
            returned branches against the pose that was asked for, because the
            solver clamps at the workspace boundary instead of failing: it can
            return a finite, in-limits configuration for a pose it misses by
            centimetres.  Genuine solutions reproduce the pose to ~1e-12.

            No joint-limit filtering beyond the solver's own (which uses PANDA
            limits) is applied — an FR3 caller must re-filter.

            Parameters
            ----------
            O_T_EE_flat : (N,16) float64
                Hand-TCP poses, each a 4x4 flattened COLUMN-major.
            q7 : (N,) float64
                Redundancy parameter (joint 7) per row.
            q_actual : (N,7) or (7,) float64
                Seed configuration; a single (7,) seed is broadcast.
            pos_tol : float
                Position tolerance of the FK verification, metres.
            rot_tol : float
                Frobenius tolerance on the rotation block.

            Returns
            -------
            (N,4,7) float64
                Branch solutions.  A branch that was NaN, or that failed the FK
                verification, is an all-NaN row.
        )pbdoc"
    );

    m.def("fk_batch",
        [](py::array q_in, double tcp) {
            auto q = as_c64(q_in);
            if (q.ndim() != 2 || q.shape(1) != 7) {
                throw std::invalid_argument("q must be (N,7)");
            }
            const py::ssize_t N = q.shape(0);
            py::array_t<double> T_out({N, py::ssize_t(4), py::ssize_t(4)});
            py::array_t<double> P_out({N, py::ssize_t(9), py::ssize_t(3)});
            const double* q_p = q.data();
            double* T_p = T_out.mutable_data();
            double* P_p = P_out.mutable_data();
            {
                py::gil_scoped_release release;
                FkChain chain;
                for (py::ssize_t i = 0; i < N; ++i) {
                    fk_chain(q_p + i * 7, tcp, chain);
                    double* Td = T_p + i * 16;
                    for (int r = 0; r < 4; ++r)
                        for (int c = 0; c < 4; ++c) Td[r * 4 + c] = chain.T(r, c);
                    double* Pd = P_p + i * 27;
                    for (int k = 0; k < 9; ++k) {
                        Pd[k * 3 + 0] = chain.p[k][0];
                        Pd[k * 3 + 1] = chain.p[k][1];
                        Pd[k * 3 + 2] = chain.p[k][2];
                    }
                }
            }
            py::dict d;
            d["T"] = T_out;
            d["pts"] = P_out;
            return d;
        },
        py::arg("q"), py::arg("tcp") = TCP_D_DEFAULT,
        R"pbdoc(
            Batched forward kinematics.

            Parameters
            ----------
            q : (N,7) float64
            tcp : float
                Hand-TCP offset along tool z after the Rz(-pi/4) flange twist.

            Returns
            -------
            dict
                "T"   (N,4,4) hand-TCP poses in link0 (row-major 4x4 each)
                "pts" (N,9,3) chain points: base origin, J1..J7 origins, TCP
        )pbdoc"
    );

    m.def("tip_jacobian_batch",
        [](py::array q_in, double pen_ext, double tcp) {
            auto q = as_c64(q_in);
            if (q.ndim() != 2 || q.shape(1) != 7) {
                throw std::invalid_argument("q must be (N,7)");
            }
            const py::ssize_t N = q.shape(0);
            py::array_t<double> J_out({N, py::ssize_t(3), py::ssize_t(7)});
            const double* q_p = q.data();
            double* J_p = J_out.mutable_data();
            {
                py::gil_scoped_release release;
                FkChain chain;
                for (py::ssize_t i = 0; i < N; ++i) {
                    fk_chain(q_p + i * 7, tcp, chain);
                    const Eigen::Vector3d tip =
                        chain.T.block<3, 1>(0, 3)
                        + chain.T.block<3, 3>(0, 0)
                              * Eigen::Vector3d(0.0, 0.0, pen_ext);
                    double* Jd = J_p + i * 21;
                    for (int j = 0; j < 7; ++j) {
                        const Eigen::Vector3d col =
                            chain.z[j].cross(tip - chain.p[j + 1]);
                        Jd[0 * 7 + j] = col[0];
                        Jd[1 * 7 + j] = col[1];
                        Jd[2 * 7 + j] = col[2];
                    }
                }
            }
            return J_out;
        },
        py::arg("q"), py::arg("pen_ext"), py::arg("tcp") = TCP_D_DEFAULT,
        R"pbdoc(
            Batched ANALYTIC geometric position Jacobian of the pen tip.

            tip = hand-TCP + pen_ext along tool z; column i is
            z_i x (p_tip - p_i) with z_i, p_i the axis and origin of joint i
            taken from the modified-DH chain.  Exact — it replaces a 14-FK
            central finite difference.

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
        )pbdoc"
    );

    m.attr("__version__") = "1.0.0";
    m.attr("has_batch") = true;   // feature flag for callers with an old wheel
}