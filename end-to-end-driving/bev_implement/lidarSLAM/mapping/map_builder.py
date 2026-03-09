import numpy as np
from slam.pose_utils import transform_points


class MapBuilder:

    def __init__(self, max_points=2000000, downsample_voxel=0.3):

        self.global_map = None
        self.max_points = int(max_points)
        self.downsample_voxel = float(downsample_voxel)

    def _voxel_downsample_numpy(self, points):
        if points is None or len(points) == 0:
            return points
        if self.downsample_voxel <= 0:
            return points

        vox = np.floor(points / self.downsample_voxel).astype(np.int64)
        _, keep = np.unique(vox, axis=0, return_index=True)
        return points[np.sort(keep)]


    def add_scan(self,scan,pose):

        pts = transform_points(scan,pose)

        if self.global_map is None:

            self.global_map = pts

        else:

            self.global_map = np.vstack((self.global_map,pts))

        if len(self.global_map) > self.max_points:
            self.global_map = self._voxel_downsample_numpy(self.global_map)


    def get_map(self):

        return self.global_map
