"""
Minimal LiDAR SLAM pipeline for Waymo:
1) Frame loader + point filtering
2) ICP odometry frontend
3) Pose graph with odometry + loop-closure edges
4) Loop detection from scan context + FPFH similarity
5) Graph optimization
6) Global map reconstruction
7) Trajectory/map visualization + ATE/RPE evaluation
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from torch_waymo import WaymoDataset

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import open3d as o3d
except ImportError:
    o3d = None

try:
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
except ImportError:
    least_squares = None
    Rotation = None


def _array_from_any(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None

    if isinstance(x, np.ndarray):
        return x

    # Torch tensor support without hard dependency.
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            pass

    try:
        arr = np.asarray(x)
        # Object arrays are usually not usable for point extraction.
        if arr.dtype == object:
            return None
        return arr
    except Exception:
        return None


def _extract_xyz_candidates(obj: Any, seen: Optional[set] = None) -> List[np.ndarray]:
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return []
    seen.add(oid)

    candidates: List[np.ndarray] = []

    arr = _array_from_any(obj)
    if arr is not None:
        if arr.ndim == 2 and arr.shape[1] >= 3:
            candidates.append(arr[:, :3])
        elif arr.ndim == 3 and arr.shape[-1] >= 3:
            candidates.append(arr.reshape(-1, arr.shape[-1])[:, :3])

    if isinstance(obj, dict):
        for k in ("xyz", "points", "point_xyz", "data"):
            if k in obj:
                candidates.extend(_extract_xyz_candidates(obj[k], seen))

    for attr in ("xyz", "points", "point_xyz", "data"):
        if hasattr(obj, attr):
            try:
                candidates.extend(_extract_xyz_candidates(getattr(obj, attr), seen))
            except Exception:
                pass

    # Some datasets expose separate x/y/z arrays.
    if hasattr(obj, "x") and hasattr(obj, "y") and hasattr(obj, "z"):
        ax = _array_from_any(getattr(obj, "x"))
        ay = _array_from_any(getattr(obj, "y"))
        az = _array_from_any(getattr(obj, "z"))
        if ax is not None and ay is not None and az is not None:
            if ax.shape == ay.shape == az.shape:
                xyz = np.stack([ax.reshape(-1), ay.reshape(-1), az.reshape(-1)], axis=1)
                candidates.append(xyz)

    if isinstance(obj, (list, tuple)):
        for item in obj:
            candidates.extend(_extract_xyz_candidates(item, seen))

    return candidates


def as_numpy_points(points: Any) -> np.ndarray:
    xyz_chunks = _extract_xyz_candidates(points)
    xyz_chunks = [c for c in xyz_chunks if c.ndim == 2 and c.shape[1] == 3 and len(c) > 0]

    if not xyz_chunks:
        raise ValueError(f"Unable to parse point cloud from frame.points (type={type(points)})")

    pts = np.concatenate(xyz_chunks, axis=0).astype(np.float64, copy=False)

    # Guard against invalid rows.
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    if len(pts) == 0:
        raise ValueError("Parsed points are empty after finite-value filtering.")
    return pts


def as_numpy_pose(pose_obj: Any) -> np.ndarray:
    if pose_obj is None:
        return np.eye(4, dtype=np.float64)
    if hasattr(pose_obj, "transform"):
        pose_arr = np.asarray(pose_obj.transform)
    else:
        pose_arr = np.asarray(pose_obj)
    pose_arr = pose_arr.astype(np.float64)
    if pose_arr.size == 16:
        return pose_arr.reshape(4, 4)
    raise ValueError(f"Unsupported pose format with size {pose_arr.size}")


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ points.T).T + t


def inv_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def se3_from_rt(Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = t
    return T


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0 or voxel_size <= 0:
        return points
    vox = np.floor(points / voxel_size).astype(np.int64)
    _, uniq_idx = np.unique(vox, axis=0, return_index=True)
    return points[np.sort(uniq_idx)]


def best_fit_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_cent = src.mean(axis=0)
    dst_cent = dst.mean(axis=0)
    src_c = src - src_cent
    dst_c = dst - dst_cent
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:
        Vt[-1, :] *= -1
        Rm = Vt.T @ U.T
    t = dst_cent - Rm @ src_cent
    return se3_from_rt(Rm, t)


def nearest_neighbor_bruteforce(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(src) == 0 or len(dst) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64)
    d2 = np.sum((src[:, None, :] - dst[None, :, :]) ** 2, axis=2)
    idx = np.argmin(d2, axis=1)
    min_d = np.sqrt(d2[np.arange(len(src)), idx])
    return idx, min_d


def icp_point_to_point(
    source: np.ndarray,
    target: np.ndarray,
    max_iters: int = 20,
    max_corr_dist: float = 2.0,
) -> Tuple[np.ndarray, float]:
    if len(source) < 20 or len(target) < 20:
        return np.eye(4, dtype=np.float64), np.inf

    src = source.copy()
    T_total = np.eye(4, dtype=np.float64)
    prev_error = np.inf

    for _ in range(max_iters):
        idx, d = nearest_neighbor_bruteforce(src, target)
        mask = d < max_corr_dist
        if mask.sum() < 20:
            break
        src_corr = src[mask]
        dst_corr = target[idx[mask]]
        T_step = best_fit_transform(src_corr, dst_corr)
        src = transform_points(src, T_step)
        T_total = T_step @ T_total
        error = float(np.mean(d[mask]))
        if abs(prev_error - error) < 1e-4:
            prev_error = error
            break
        prev_error = error
    return T_total, prev_error


def build_scan_context(
    points: np.ndarray,
    num_rings: int = 20,
    num_sectors: int = 60,
    max_radius: float = 80.0,
) -> np.ndarray:
    sc = np.zeros((num_rings, num_sectors), dtype=np.float64)
    if len(points) == 0:
        return sc

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2)
    theta = (np.arctan2(y, x) + 2.0 * np.pi) % (2.0 * np.pi)
    ring_idx = np.floor((r / max_radius) * num_rings).astype(np.int32)
    sector_idx = np.floor((theta / (2.0 * np.pi)) * num_sectors).astype(np.int32)

    valid = (ring_idx >= 0) & (ring_idx < num_rings) & (r <= max_radius)
    ring_idx = ring_idx[valid]
    sector_idx = np.clip(sector_idx[valid], 0, num_sectors - 1)
    z_valid = z[valid]

    for ri, si, zi in zip(ring_idx, sector_idx, z_valid):
        if zi > sc[ri, si]:
            sc[ri, si] = zi
    return sc


def scan_context_similarity(sc_a: np.ndarray, sc_b: np.ndarray) -> float:
    num_sectors = sc_a.shape[1]
    best = -1.0
    a = sc_a.reshape(-1)
    na = np.linalg.norm(a) + 1e-9
    for shift in range(num_sectors):
        b_shift = np.roll(sc_b, shift=shift, axis=1).reshape(-1)
        nb = np.linalg.norm(b_shift) + 1e-9
        sim = float((a @ b_shift) / (na * nb))
        if sim > best:
            best = sim
    return max(best, 0.0)


def compute_fpfh_descriptor(points: np.ndarray, voxel_size: float = 0.5) -> Optional[np.ndarray]:
    if o3d is None or len(points) < 200:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=max(voxel_size, 0.1))
    if len(pcd.points) < 50:
        return None
    radius_normal = voxel_size * 2.0
    radius_feature = voxel_size * 5.0
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    hist = np.asarray(fpfh.data)  # [33, N]
    if hist.size == 0:
        return None
    desc = hist.mean(axis=1)
    desc /= np.linalg.norm(desc) + 1e-9
    return desc


def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def pose_to_vec6(T: np.ndarray) -> np.ndarray:
    if Rotation is None:
        raise RuntimeError("scipy is required for pose graph optimization")
    rot = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    trans = T[:3, 3]
    return np.concatenate([trans, rot], axis=0)


def vec6_to_pose(v: np.ndarray) -> np.ndarray:
    if Rotation is None:
        raise RuntimeError("scipy is required for pose graph optimization")
    t = v[:3]
    r = Rotation.from_rotvec(v[3:]).as_matrix()
    return se3_from_rt(r, t)


def se3_log_residual(T_err: np.ndarray) -> np.ndarray:
    if Rotation is None:
        raise RuntimeError("scipy is required for pose graph optimization")
    trans = T_err[:3, 3]
    rotvec = Rotation.from_matrix(T_err[:3, :3]).as_rotvec()
    return np.concatenate([trans, rotvec], axis=0)


@dataclass
class PoseGraphEdge:
    i: int
    j: int
    Tij: np.ndarray
    weight: float
    edge_type: str  # "odom" or "loop"


@dataclass
class Keyframe:
    node_id: int
    scan: np.ndarray
    scan_context: np.ndarray
    fpfh: Optional[np.ndarray]


class PoseGraph:
    def __init__(self) -> None:
        self.nodes: List[np.ndarray] = []
        self.edges: List[PoseGraphEdge] = []

    def add_node(self, pose: np.ndarray) -> int:
        self.nodes.append(pose.copy())
        return len(self.nodes) - 1

    def add_edge(self, i: int, j: int, Tij: np.ndarray, weight: float, edge_type: str) -> None:
        self.edges.append(PoseGraphEdge(i=i, j=j, Tij=Tij.copy(), weight=weight, edge_type=edge_type))

    def optimize(self, max_nfev: int = 50) -> None:
        if least_squares is None or Rotation is None:
            print("Pose graph optimization skipped (scipy unavailable).")
            return
        if len(self.nodes) <= 2 or len(self.edges) == 0:
            return

        x0 = np.concatenate([pose_to_vec6(T) for T in self.nodes[1:]], axis=0)

        def unpack(x: np.ndarray) -> List[np.ndarray]:
            poses = [self.nodes[0].copy()]
            for k in range(len(self.nodes) - 1):
                poses.append(vec6_to_pose(x[k * 6 : (k + 1) * 6]))
            return poses

        def residual_fn(x: np.ndarray) -> np.ndarray:
            poses = unpack(x)
            residuals: List[np.ndarray] = []
            for e in self.edges:
                Ti = poses[e.i]
                Tj = poses[e.j]
                Tij_pred = inv_se3(Ti) @ Tj
                Terr = inv_se3(e.Tij) @ Tij_pred
                r = se3_log_residual(Terr) * np.sqrt(max(e.weight, 1e-6))
                residuals.append(r)
            return np.concatenate(residuals, axis=0)

        out = least_squares(residual_fn, x0, method="lm", max_nfev=max_nfev)
        x_opt = out.x
        for k in range(1, len(self.nodes)):
            self.nodes[k] = vec6_to_pose(x_opt[(k - 1) * 6 : k * 6])
        print(f"Pose graph optimized: success={out.success}, cost={out.cost:.4f}, iters={out.nfev}")


class WaymoSLAMLoader:
    def __init__(self, dataset_path: str, split: str = "training") -> None:
        self.dataset = WaymoDataset(dataset_path, split)

    def __len__(self) -> int:
        return len(self.dataset)

    def get(self, i: int) -> Dict[str, Any]:
        frame = self.dataset[i]
        points = as_numpy_points(frame.points)
        pose = as_numpy_pose(getattr(frame, "pose", None))
        ts = int(getattr(frame, "timestamp_micros", i))
        labels = getattr(frame, "laser_labels", None)
        return {
            "points": points,
            "pose": pose,
            "timestamp_micros": ts,
            "laser_labels": labels,
        }


class LidarSLAMPipeline:
    def __init__(
        self,
        loader: WaymoSLAMLoader,
        max_frames: int = 200,
        keyframe_every: int = 5,
        voxel_size: float = 0.3,
        icp_corr_dist: float = 2.0,
        loop_min_separation: int = 40,
        loop_sc_threshold: float = 0.80,
        loop_fpfh_threshold: float = 0.75,
        loop_combined_threshold: float = 0.78,
    ) -> None:
        self.loader = loader
        self.max_frames = max_frames
        self.keyframe_every = max(1, keyframe_every)
        self.voxel_size = voxel_size
        self.icp_corr_dist = icp_corr_dist
        self.loop_min_separation = loop_min_separation
        self.loop_sc_threshold = loop_sc_threshold
        self.loop_fpfh_threshold = loop_fpfh_threshold
        self.loop_combined_threshold = loop_combined_threshold

        self.pose_graph = PoseGraph()
        self.keyframes: List[Keyframe] = []

        self.scans_filtered: List[np.ndarray] = []
        self.gt_poses: List[np.ndarray] = []

    def filter_points(
        self,
        points: np.ndarray,
        min_range: float = 2.0,
        max_range: float = 80.0,
        min_z: float = -3.0,
        max_z: float = 3.0,
    ) -> np.ndarray:
        r = np.linalg.norm(points[:, :2], axis=1)
        mask = (r > min_range) & (r < max_range) & (points[:, 2] > min_z) & (points[:, 2] < max_z)
        return points[mask]

    def filter_dynamic_objects(self, points: np.ndarray, labels: Any, speed_threshold: float = 1.0) -> np.ndarray:
        if labels is None:
            return points
        keep = np.ones(len(points), dtype=bool)
        for obj in labels:
            box = getattr(obj, "box", None)
            if box is None:
                continue
            sx = float(getattr(getattr(obj, "metadata", None), "speed_x", 0.0))
            sy = float(getattr(getattr(obj, "metadata", None), "speed_y", 0.0))
            speed = np.sqrt(sx * sx + sy * sy)
            if speed < speed_threshold:
                continue
            cx, cy, cz = float(box.center_x), float(box.center_y), float(box.center_z)
            lx, ly, lz = float(box.length), float(box.width), float(box.height)
            yaw = float(box.heading)
            Rz = np.array(
                [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            p_local = (Rz.T @ (points - np.array([cx, cy, cz], dtype=np.float64)).T).T
            inside = (
                (np.abs(p_local[:, 0]) <= lx * 0.5)
                & (np.abs(p_local[:, 1]) <= ly * 0.5)
                & (np.abs(p_local[:, 2]) <= lz * 0.5)
            )
            keep &= ~inside
        return points[keep]

    def add_keyframe(self, node_id: int, scan: np.ndarray) -> None:
        sc = build_scan_context(scan)
        fpfh = compute_fpfh_descriptor(scan, voxel_size=max(self.voxel_size, 0.2))
        self.keyframes.append(Keyframe(node_id=node_id, scan=scan, scan_context=sc, fpfh=fpfh))

    def detect_loop(self, curr_kf: Keyframe) -> Optional[Tuple[int, float]]:
        best_node = None
        best_score = -1.0
        for cand in self.keyframes[:-1]:
            if curr_kf.node_id - cand.node_id < self.loop_min_separation:
                continue
            sc_sim = scan_context_similarity(curr_kf.scan_context, cand.scan_context)
            fpfh_sim = cosine_similarity(curr_kf.fpfh, cand.fpfh)
            combined = 0.7 * sc_sim + 0.3 * fpfh_sim
            if sc_sim < self.loop_sc_threshold:
                continue
            if cand.fpfh is not None and curr_kf.fpfh is not None and fpfh_sim < self.loop_fpfh_threshold:
                continue
            if combined > best_score:
                best_score = combined
                best_node = cand.node_id
        if best_node is None or best_score < self.loop_combined_threshold:
            return None
        return best_node, best_score

    def run(self) -> None:
        n = min(len(self.loader), self.max_frames)
        if n <= 0:
            print("No frames found.")
            return

        prev_scan = None
        prev_node = None

        for i in range(n):
            item = self.loader.get(i)
            points = item["points"]
            gt_pose = item["pose"]
            labels = item["laser_labels"]

            points = self.filter_points(points)
            points = self.filter_dynamic_objects(points, labels)
            points = voxel_downsample(points, self.voxel_size)

            self.scans_filtered.append(points)
            self.gt_poses.append(gt_pose)

            if i == 0:
                node = self.pose_graph.add_node(np.eye(4, dtype=np.float64))
                self.add_keyframe(node, points)
                prev_scan = points
                prev_node = node
                continue

            T_rel, icp_err = icp_point_to_point(
                source=points,
                target=prev_scan,
                max_iters=20,
                max_corr_dist=self.icp_corr_dist,
            )
            curr_pose = self.pose_graph.nodes[prev_node] @ T_rel
            curr_node = self.pose_graph.add_node(curr_pose)

            odom_weight = 5.0 if np.isfinite(icp_err) else 1.0
            self.pose_graph.add_edge(prev_node, curr_node, T_rel, odom_weight, edge_type="odom")

            if i % self.keyframe_every == 0:
                self.add_keyframe(curr_node, points)
                curr_kf = self.keyframes[-1]
                loop = self.detect_loop(curr_kf)
                if loop is not None:
                    loop_node, score = loop
                    scan_loop = self.scans_filtered[loop_node]
                    T_loop, loop_err = icp_point_to_point(
                        source=points,
                        target=scan_loop,
                        max_iters=30,
                        max_corr_dist=self.icp_corr_dist * 1.25,
                    )
                    if np.isfinite(loop_err) and loop_err < 1.5:
                        self.pose_graph.add_edge(loop_node, curr_node, T_loop, 15.0, edge_type="loop")
                        print(f"Loop edge added: X{loop_node} ---- X{curr_node}  score={score:.3f}, icp={loop_err:.3f}")

            prev_scan = points
            prev_node = curr_node

        self.pose_graph.optimize()

    def reconstruct_global_map(self, map_voxel_size: float = 0.2, downsample_every: int = 20) -> np.ndarray:
        global_pts: List[np.ndarray] = []
        for i, scan in enumerate(self.scans_filtered):
            pose_i = self.pose_graph.nodes[i]
            transformed = transform_points(scan, pose_i)
            global_pts.append(transformed)
            if (i + 1) % downsample_every == 0:
                merged = np.concatenate(global_pts, axis=0)
                merged = voxel_downsample(merged, map_voxel_size)
                global_pts = [merged]
        if len(global_pts) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        global_map = np.concatenate(global_pts, axis=0)
        return voxel_downsample(global_map, map_voxel_size)

    def evaluate_trajectory(self) -> Dict[str, float]:
        if len(self.gt_poses) != len(self.pose_graph.nodes) or len(self.gt_poses) < 2:
            return {"ate_rmse": float("nan"), "rpe_trans_rmse": float("nan"), "rpe_rot_deg_rmse": float("nan")}

        est = self.pose_graph.nodes
        gt = self.gt_poses

        align = gt[0] @ inv_se3(est[0])
        est_aligned = [align @ p for p in est]

        ate_sq = []
        rpe_t_sq = []
        rpe_r_sq = []
        for i in range(len(est_aligned)):
            te = np.linalg.norm(est_aligned[i][:3, 3] - gt[i][:3, 3])
            ate_sq.append(te * te)

            if i == 0:
                continue
            d_est = inv_se3(est_aligned[i - 1]) @ est_aligned[i]
            d_gt = inv_se3(gt[i - 1]) @ gt[i]
            d_err = inv_se3(d_gt) @ d_est
            t_err = np.linalg.norm(d_err[:3, 3])
            r_err = 0.0
            if Rotation is not None:
                r_err = np.linalg.norm(Rotation.from_matrix(d_err[:3, :3]).as_rotvec())
            rpe_t_sq.append(t_err * t_err)
            rpe_r_sq.append(np.degrees(r_err) ** 2)

        return {
            "ate_rmse": float(np.sqrt(np.mean(ate_sq))),
            "rpe_trans_rmse": float(np.sqrt(np.mean(rpe_t_sq))) if rpe_t_sq else float("nan"),
            "rpe_rot_deg_rmse": float(np.sqrt(np.mean(rpe_r_sq))) if rpe_r_sq else float("nan"),
        }

    def plot_trajectory(self) -> None:
        if plt is None:
            print("matplotlib unavailable, skipping trajectory plot.")
            return
        est_xy = np.array([p[:3, 3] for p in self.pose_graph.nodes])
        plt.figure(figsize=(8, 6))
        plt.plot(est_xy[:, 0], est_xy[:, 1], label="Estimated")
        if len(self.gt_poses) == len(self.pose_graph.nodes):
            gt_xy = np.array([p[:3, 3] for p in self.gt_poses])
            plt.plot(gt_xy[:, 0], gt_xy[:, 1], label="Ground Truth", alpha=0.7)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("SLAM Trajectory")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.show()

    def visualize_map_open3d(self, global_map: np.ndarray) -> None:
        if o3d is None:
            print("Open3D unavailable, skipping map visualization.")
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(global_map.astype(np.float64))
        o3d.visualization.draw_geometries([pcd])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Waymo LiDAR SLAM baseline with loop closure.")
    parser.add_argument("--dataset-path", type=str, default="/workdir/data/pytorch-e2e/converted/")
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--keyframe-every", type=int, default=5)
    parser.add_argument("--voxel-size", type=float, default=0.3)
    parser.add_argument("--icp-corr-dist", type=float, default=2.0)
    parser.add_argument("--loop-min-separation", type=int, default=40)
    parser.add_argument("--loop-sc-threshold", type=float, default=0.80)
    parser.add_argument("--loop-fpfh-threshold", type=float, default=0.75)
    parser.add_argument("--loop-combined-threshold", type=float, default=0.78)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-vis-map", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = WaymoSLAMLoader(dataset_path=args.dataset_path, split=args.split)
    pipeline = LidarSLAMPipeline(
        loader=loader,
        max_frames=args.max_frames,
        keyframe_every=args.keyframe_every,
        voxel_size=args.voxel_size,
        icp_corr_dist=args.icp_corr_dist,
        loop_min_separation=args.loop_min_separation,
        loop_sc_threshold=args.loop_sc_threshold,
        loop_fpfh_threshold=args.loop_fpfh_threshold,
        loop_combined_threshold=args.loop_combined_threshold,
    )
    pipeline.run()
    global_map = pipeline.reconstruct_global_map()
    metrics = pipeline.evaluate_trajectory()

    print("Evaluation:")
    print(f"  ATE RMSE:        {metrics['ate_rmse']:.4f} m")
    print(f"  RPE trans RMSE:  {metrics['rpe_trans_rmse']:.4f} m")
    print(f"  RPE rot RMSE:    {metrics['rpe_rot_deg_rmse']:.4f} deg")
    print(f"Global map points: {len(global_map)}")

    if not args.no_plot:
        pipeline.plot_trajectory()
    if not args.no_vis_map:
        pipeline.visualize_map_open3d(global_map)


if __name__ == "__main__":
    main()
