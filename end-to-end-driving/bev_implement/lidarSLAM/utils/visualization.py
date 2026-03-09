import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def plot_trajectory(poses, save_path="trajectory.png", show=False):

    xy = np.array([p[:3,3] for p in poses])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xy[:,0], xy[:,1], linewidth=1.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Estimated trajectory")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        # Non-blocking show to avoid hanging script execution.
        plt.show(block=False)
        plt.pause(0.1)

    plt.close(fig)


def visualize_map(points, show=True):
    if not show:
        return

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.visualization.draw_geometries([pcd])
