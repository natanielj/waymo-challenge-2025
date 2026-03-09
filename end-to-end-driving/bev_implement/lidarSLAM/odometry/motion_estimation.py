import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares

from slam.pose_utils import transform_points, vec6_to_pose


def edge_residual(p, a, b):

    ab = b-a

    ap = p-a

    return np.linalg.norm(np.cross(ap,ab))/(np.linalg.norm(ab)+1e-6)


def plane_residual(p,a,b,c):

    normal = np.cross(b-a,c-a)

    normal /= np.linalg.norm(normal)+1e-6

    return np.dot(p-a,normal)


def estimate_motion(edges, planes, prev_edges, prev_planes):
    edges = np.asarray(edges)
    planes = np.asarray(planes)
    prev_edges = np.asarray(prev_edges)
    prev_planes = np.asarray(prev_planes)

    if len(edges) == 0 or len(planes) == 0 or len(prev_edges) < 2 or len(prev_planes) < 3:
        return np.eye(4)

    edge_tree = cKDTree(prev_edges)
    plane_tree = cKDTree(prev_planes)

    edge_corr = []
    for p in edges:
        d, idx = edge_tree.query(p, k=2)
        if np.isfinite(d[1]) and d[1] < 1.0:
            a, b = prev_edges[idx]
            edge_corr.append((p, a, b))

    plane_corr = []
    for p in planes:
        d, idx = plane_tree.query(p, k=3)
        if np.isfinite(d[2]) and d[2] < 1.0:
            a, b, c = prev_planes[idx]
            plane_corr.append((p, a, b, c))

    # Levenberg-Marquardt requires a fixed residual size and enough equations.
    if len(edge_corr) + len(plane_corr) < 6:
        return np.eye(4)

    def residual(x):
        T = vec6_to_pose(x)
        res = np.empty(len(edge_corr) + len(plane_corr), dtype=np.float64)
        i = 0

        for p, a, b in edge_corr:
            pt = transform_points(p[None], T)[0]
            res[i] = edge_residual(pt, a, b)
            i += 1

        for p, a, b, c in plane_corr:
            pt = transform_points(p[None], T)[0]
            res[i] = plane_residual(pt, a, b, c)
            i += 1

        return res

    x0 = np.zeros(6)
    result = least_squares(residual, x0, method="lm", max_nfev=20)
    return vec6_to_pose(result.x)
