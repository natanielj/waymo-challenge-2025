import numpy as np

from dataset.loader import WaymoLoader

from preprocessing.filtering import filter_points, voxel_downsample

from features.feature_extraction import extract_features

from odometry.motion_estimation import estimate_motion

from slam.pose_graph import PoseGraph

from mapping.map_builder import MapBuilder

from utils.visualization import plot_trajectory, visualize_map

from mapping.bev_generator import BEVGenerator
from slam.pose_utils import transform_points


def main(dataset_path):
    use_ground_truth_pose = True
    max_translation_step_m = 2.5

    loader = WaymoLoader(dataset_path)

    pose_graph = PoseGraph()

    map_builder = MapBuilder()


    prev_edges = None

    prev_planes = None

    prev_pose = None

    bev = BEVGenerator(
        resolution=0.2,
        x_range=(-80,80),
        y_range=(-80,80)
    )
    bev_origin = None

    for i in range(200):

        frame = loader.get_frame(i)

        points = frame["points"]

        points = filter_points(points)

        points = voxel_downsample(points,0.3)


        edges,planes = extract_features(points)
        if i == 0:

            pose = frame["pose"]
            pose_graph.add_node(pose)
            bev_origin = pose[:3, 3].copy()

            prev_edges = edges

            prev_planes = planes

            prev_pose = pose
        else:
            if use_ground_truth_pose:
                pose = frame["pose"]
                T = np.linalg.inv(prev_pose) @ pose
            else:
                T = estimate_motion(edges,planes,prev_edges,prev_planes)
                step = float(np.linalg.norm(T[:3, 3]))
                if step > max_translation_step_m:
                    print("large odometry step, clamping at frame", i, "step[m]=", round(step, 3))
                    T = np.eye(4, dtype=np.float32)
                pose = prev_pose @ T
            pose_graph.add_node(pose)


        prev_edges = edges

        prev_planes = planes

        prev_pose = pose

        map_builder.add_scan(points,pose)
        # BEV projection uses near-ground points to suppress ghost trails.
        bev_points = points[(points[:, 2] > -2.0) & (points[:, 2] < 1.0)]
        transformed = transform_points(bev_points, pose)
        if bev_origin is not None:
            transformed = transformed - bev_origin[None, :]
        bev.add_points(transformed)


        print("frame", i, "step[m]=", round(float(np.linalg.norm(T[:3, 3])), 3) if i > 0 else 0.0)


    plot_trajectory(pose_graph.nodes, save_path="trajectory.png", show=False)

    global_map = map_builder.get_map()

    bev_map = bev.get_bev()
    print("BEV points in-range:", bev.in_range_points, "/", bev.total_points)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,8))
    plt.imshow(bev_map, cmap="gray", origin="lower")
    plt.title("BEV Map")
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

    plt.imsave("bev_map.png", bev_map, cmap="gray")

    
    visualize_map(global_map, show=False)


if __name__ == "__main__":

    main("/workdir/data/pytorch-e2e/converted/")
