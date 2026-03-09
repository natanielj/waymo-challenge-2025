import numpy as np


class BEVGenerator:

    def __init__(self, resolution=0.2, x_range=(-80,80), y_range=(-80,80)):

        self.resolution = resolution

        self.x_range = x_range
        self.y_range = y_range

        self.width = int((x_range[1] - x_range[0]) / resolution)
        self.height = int((y_range[1] - y_range[0]) / resolution)

        self.grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.total_points = 0
        self.in_range_points = 0


    def add_points(self, points):
        if points is None or len(points) == 0:
            return

        self.total_points += len(points)
        x = points[:,0]
        y = points[:,1]

        ix = ((x - self.x_range[0]) / self.resolution).astype(np.int32)
        iy = ((y - self.y_range[0]) / self.resolution).astype(np.int32)

        valid = (
            (ix >= 0) & (ix < self.width) &
            (iy >= 0) & (iy < self.height)
        )

        ix = ix[valid]
        iy = iy[valid]
        self.in_range_points += len(ix)

        self.grid[iy, ix] += 1


    def get_bev(self):

        bev = np.log1p(self.grid)
        max_val = bev.max()
        if max_val > 0:
            bev = bev / max_val

        return bev
