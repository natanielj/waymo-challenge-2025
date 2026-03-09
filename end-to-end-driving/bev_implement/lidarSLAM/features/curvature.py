import numpy as np


def compute_curvature(points):

    N = len(points)

    curvature = np.zeros(N)

    for i in range(5, N-5):

        diff = np.sum(points[i-5:i+5], axis=0) - 10*points[i]

        curvature[i] = np.linalg.norm(diff)**2

    return curvature