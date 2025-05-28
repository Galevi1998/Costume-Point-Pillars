import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def create_lidar_pointcloud(points):
    z_values = points[:, 2]
    z_min, z_max = z_values.min(), z_values.max()
    normalized_z = (z_values - z_min) / (z_max - z_min + 1e-5)
    colormap = plt.get_cmap('viridis')
    colors = colormap(normalized_z)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def create_3d_box(center, size, yaw, color=[1, 0, 0]):
    l, w, h = size
    corners = np.array([
        [ l/2,  w/2, 0], [ l/2, -w/2, 0], [-l/2, -w/2, 0], [-l/2,  w/2, 0],
        [ l/2,  w/2, h], [ l/2, -w/2, h], [-l/2, -w/2, h], [-l/2,  w/2, h],
    ])
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    rotated = corners @ R.T + np.array(center)

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    colors = [color for _ in lines]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(rotated)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def classify_size(size):
    return tuple(np.round(size, 2))



def generate_color(index, total=20):
    cmap = plt.get_cmap("tab20")
    return list(cmap(index % total)[:3])


def main():
    id = "00001"
    path = f"/lidar3d_detection_ws/training/training/velodyne/{id}.bin"
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    pcd = create_lidar_pointcloud(points)

    bboxes = np.array([
        [18.29, -13.23, 1.29, 2.36, 2.36, 4.36, 0.0],
        [11.34, -4.20, 1.01, 2.36, 2.36, 4.36, -3.12],
        [16.46, -3.72, 1.06, 2.36, 2.36, 4.36, 3.12],
        [21.66, -3.44, 1.08, 2.36, 2.33, 4.36, 3.12],
        [5.10, -4.39, 0.98, 2.36, 2.36, 4.36, -3.12],
        [7.92, -13.87, 1.24, 2.36, 2.36, 4.36, 0.0],
        [9.02, -1.09, 0.91, 1.2, 2.0, 1.2, 3.12],
        [15.96, -14.37, 1.39, 1.2, 2.0, 1.2, -3.05],
        [27.49, -12.88, 1.41, 2.36, 2.36, 4.36, 0.03],
        [26.41, -3.19, 1.05, 2.36, 2.36, 4.36, 3.14],
        [12.73, -10.05, 1.15, 2.36, 2.36, 4.36, 0.017],
        [26.07, -9.81, 1.08, 2.36, 2.36, 4.36, 0.0087],
        [8.96, 0.36, 0.97, 1.2, 2.0, 1.2, 0.0]
    ])

    combo_to_color = {}
    boxes = []

    for i, box in enumerate(bboxes):
        center, size, yaw = box[:3], box[3:6], box[6]
        combo_key = classify_size(size)
        if combo_key not in combo_to_color:
            combo_to_color[combo_key] = generate_color(len(combo_to_color))
        color = combo_to_color[combo_key]
        boxes.append(create_3d_box(center, size, yaw, color))

    o3d.visualization.draw_geometries([pcd, *boxes])


if __name__ == "__main__":
    main()
