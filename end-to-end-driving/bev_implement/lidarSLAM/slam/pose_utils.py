import numpy as np
from scipy.spatial.transform import Rotation


def transform_points(points, T):

    R = T[:3,:3]

    t = T[:3,3]

    return (R @ points.T).T + t


def vec6_to_pose(v):

    t = v[:3]

    R = Rotation.from_rotvec(v[3:]).as_matrix()

    T = np.eye(4)

    T[:3,:3] = R

    T[:3,3] = t

    return T


def pose_to_vec6(T):

    rot = Rotation.from_matrix(T[:3,:3]).as_rotvec()

    t = T[:3,3]

    return np.concatenate([t,rot])