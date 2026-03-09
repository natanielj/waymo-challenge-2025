import numpy as np
import open3d as o3d


def filter_points(points):

    r = np.linalg.norm(points[:, :2], axis=1)

    mask = (r > 2) & (r < 80) & (points[:,2] > -1.5)

    return points[mask]


def voxel_downsample(points, voxel=0.3):

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)

    pcd = pcd.voxel_down_sample(voxel)

    return np.asarray(pcd.points)