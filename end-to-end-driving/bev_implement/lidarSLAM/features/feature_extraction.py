import numpy as np
from .curvature import compute_curvature


def extract_features(points):

    curvature = compute_curvature(points)

    idx = np.argsort(curvature)

    n = len(points)

    edge_idx = idx[-int(n*0.02):]

    plane_idx = idx[:int(n*0.04)]

    edges = points[edge_idx]

    planes = points[plane_idx]

    return edges, planes