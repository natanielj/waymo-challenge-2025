import numpy as np
from torch_waymo import WaymoDataset


class WaymoLoader:

    def __init__(self, dataset_path, split="training"):
        self.dataset = WaymoDataset(dataset_path, split)

    def __len__(self):
        return len(self.dataset)

    def _to_numpy(self, x):
        if x is None:
            return None
        if isinstance(x, (list, tuple, dict)):
            return None
        if isinstance(x, np.ndarray):
            return x
        if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
            return x.detach().cpu().numpy()
        try:
            arr = np.asarray(x)
            if arr.dtype == object:
                return None
            return arr
        except Exception:
            return None

    def _collect_xyz(self, obj, seen=None):
        if seen is None:
            seen = set()
        oid = id(obj)
        if oid in seen:
            return []
        seen.add(oid)

        xyz_chunks = []
        arr = self._to_numpy(obj)
        if arr is not None:
            if arr.ndim == 2 and arr.shape[1] >= 3:
                xyz_chunks.append(arr[:, :3])
            elif arr.ndim == 3 and arr.shape[-1] >= 3:
                xyz_chunks.append(arr.reshape(-1, arr.shape[-1])[:, :3])

        if isinstance(obj, dict):
            for key in ("xyz", "points", "point_xyz", "data"):
                if key in obj:
                    xyz_chunks.extend(self._collect_xyz(obj[key], seen))

        for attr in ("xyz", "points", "point_xyz", "data"):
            if hasattr(obj, attr):
                xyz_chunks.extend(self._collect_xyz(getattr(obj, attr), seen))

        if isinstance(obj, (list, tuple)):
            for item in obj:
                xyz_chunks.extend(self._collect_xyz(item, seen))

        return xyz_chunks

    def _parse_points(self, pts):
        xyz_chunks = self._collect_xyz(pts)
        xyz_chunks = [c for c in xyz_chunks if c.ndim == 2 and c.shape[1] == 3 and len(c) > 0]
        if not xyz_chunks:
            raise ValueError(f"Unable to parse point cloud from frame.points (type={type(pts)})")

        points = np.concatenate(xyz_chunks, axis=0).astype(np.float32, copy=False)
        valid = np.isfinite(points).all(axis=1)
        points = points[valid]
        if len(points) == 0:
            raise ValueError("Parsed points are empty after finite-value filtering.")
        return points

    def _parse_pose(self, pose):
        if pose is None:
            return np.eye(4, dtype=np.float32)
        if hasattr(pose, "transform"):
            pose_arr = np.asarray(pose.transform)
        else:
            pose_arr = np.asarray(pose)
        if pose_arr.size != 16:
            raise ValueError(f"Unsupported pose format with size {pose_arr.size}")
        return pose_arr.reshape(4, 4).astype(np.float32)

    def get_frame(self, i):

        frame = self.dataset[i]

        pts = frame.points

        points = self._parse_points(pts)
        pose = self._parse_pose(getattr(frame, "pose", None))

        return {
            "points": points,
            "pose": pose,
            "timestamp": int(getattr(frame, "timestamp_micros", i))
        }
