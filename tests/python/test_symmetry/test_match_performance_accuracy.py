"""Benchmark and accuracy checks for symmetry point matching."""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, "src/python")

from cpgeo.cpgeomodel import CPGEO


def load_case_data(npz_path: Path):
    data = np.load(npz_path)
    v = np.asarray(data["control_points"], dtype=np.float64)
    f = np.asarray(data["cp_faces"], dtype=np.int64)
    return v, f


def nearest_residual(query: np.ndarray, ref: np.ndarray, match_idx: np.ndarray):
    matched = ref[match_idx]
    d = np.linalg.norm(query - matched, axis=1)
    return float(np.max(d)), float(np.mean(d)), float(np.sqrt(np.mean(d * d)))


def nearest_optimal_dist(query: np.ndarray, ref: np.ndarray):
    diff = query[:, None, :] - ref[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return np.min(dist, axis=1)


def axial_match_error(vertices: np.ndarray, match: np.ndarray, plane: str = "yz", plane_offset: float = 0.0):
    if plane == "yz":
        n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis_id = 0
    elif plane == "xz":
        n = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis_id = 1
    elif plane == "xy":
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis_id = 2
    else:
        raise ValueError("Invalid plane")

    shift = np.zeros(3, dtype=np.float64)
    shift[axis_id] = float(plane_offset)
    r = np.eye(3, dtype=np.float64) - 2.0 * np.outer(n, n)

    v_ref = ((vertices - shift[None, :]) @ r.T) + shift[None, :]
    return nearest_residual(v_ref, vertices, match[:, 1])


def rotational_match_error(vertices: np.ndarray, match: np.ndarray, periods: int):
    alpha = 2.0 * np.pi / float(periods)
    errs = []
    for k in range(periods):
        c = float(np.cos(-alpha * k))
        s = float(np.sin(-alpha * k))
        r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        q = vertices @ r.T
        errs.append(nearest_residual(q, vertices, match[:, k]))

    max_err = max(e[0] for e in errs)
    mean_err = float(np.mean([e[1] for e in errs]))
    rmse_err = float(np.mean([e[2] for e in errs]))
    return max_err, mean_err, rmse_err


def run_case(name: str, vertices: np.ndarray, faces: np.ndarray, periods: int = 4):
    model = CPGEO(vertices, faces)

    t0 = time.perf_counter()
    v_ax, f_ax, m_ax = model.reconstruct_symmetry(
        mode="axial",
        plane="yz",
        keep_positive=True,
        tol=1e-8,
        inplace=False,
    )
    dt_ax = time.perf_counter() - t0

    if model.symmetry_points_match is None or model.symmetry_points_match.shape != m_ax.shape:
        raise RuntimeError("axial match was not stored in model.symmetry_points_match")

    ax_max, ax_mean, ax_rmse = axial_match_error(v_ax, m_ax, plane="yz", plane_offset=0.0)
    n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    r = np.eye(3, dtype=np.float64) - 2.0 * np.outer(n, n)
    q_ax = v_ax @ r.T
    d_ax_match = np.linalg.norm(q_ax - v_ax[m_ax[:, 1]], axis=1)
    d_ax_best = nearest_optimal_dist(q_ax, v_ax)
    ax_gap = d_ax_match - d_ax_best

    t1 = time.perf_counter()
    v_rot, f_rot, m_rot = model.reconstruct_symmetry(
        mode="rotational",
        periods=periods,
        tol=1e-8,
        inplace=False,
    )
    dt_rot = time.perf_counter() - t1

    if model.symmetry_points_match is None or model.symmetry_points_match.shape != m_rot.shape:
        raise RuntimeError("rotational match was not stored in model.symmetry_points_match")

    rot_max, rot_mean, rot_rmse = rotational_match_error(v_rot, m_rot, periods=periods)
    alpha = 2.0 * np.pi / float(periods)
    rot_gap_max = 0.0
    rot_gap_mean = 0.0
    for k in range(periods):
        c = float(np.cos(-alpha * k))
        s = float(np.sin(-alpha * k))
        rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        q = v_rot @ rz.T
        d_match = np.linalg.norm(q - v_rot[m_rot[:, k]], axis=1)
        d_best = nearest_optimal_dist(q, v_rot)
        gap = d_match - d_best
        rot_gap_max = max(rot_gap_max, float(np.max(gap)))
        rot_gap_mean += float(np.mean(gap))
    rot_gap_mean /= float(periods)

    print("\n" + "=" * 72)
    print(f"Case: {name}")
    print("=" * 72)
    print(f"axial    time: {dt_ax:.4f}s, max/mean/rmse error: {ax_max:.3e} / {ax_mean:.3e} / {ax_rmse:.3e}")
    print(f"rotate C{periods} time: {dt_rot:.4f}s, max/mean/rmse error: {rot_max:.3e} / {rot_mean:.3e} / {rot_rmse:.3e}")
    print(f"axial    match gap max/mean: {np.max(ax_gap):.3e} / {np.mean(ax_gap):.3e}")
    print(f"rotate C{periods} match gap max/mean: {rot_gap_max:.3e} / {rot_gap_mean:.3e}")
    print(f"axial mesh: V={v_ax.shape[0]}, F={f_ax.shape[0]}, match={m_ax.shape}")
    print(f"rot   mesh: V={v_rot.shape[0]}, F={f_rot.shape[0]}, match={m_rot.shape}")

    # Correspondence quality: chosen match should be the global nearest neighbor.
    if float(np.max(ax_gap)) > 1e-10:
        raise RuntimeError(f"{name}: axial match is not nearest-neighbor optimal, gap={np.max(ax_gap):.3e}")
    if rot_gap_max > 1e-10:
        raise RuntimeError(f"{name}: rotational match is not nearest-neighbor optimal, gap={rot_gap_max:.3e}")


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"

    v_axis, f_axis = load_case_data(data_dir / "axis.npz")
    run_case("axis_data", v_axis, f_axis, periods=4)

    v_rot, f_rot = load_case_data(data_dir / "rotation.npz")
    run_case("rotation_data", v_rot, f_rot, periods=3)

    print("\nAll match performance/accuracy checks passed.")
