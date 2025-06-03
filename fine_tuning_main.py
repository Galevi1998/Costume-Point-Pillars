import os
import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import time
import subprocess

# ------------------------------#
# Environment Setup
# ------------------------------#
os.environ["LIBGL_ALWAYS_INDIRECT"] = "1"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

# ------------------------------#
# Project Setup
# ------------------------------#
# REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPO_ROOT = "/home/user/deep_learning/Costume-Point-Pillars/datasets/INNOVIZ/"
# KITTI_DATA_PATH = os.path.join(REPO_ROOT, "training", "training")
KITTI_DATA_PATH = os.path.join(REPO_ROOT, "training")
sys.path.append(REPO_ROOT)

VELODYNE_DIR = os.path.join(KITTI_DATA_PATH, "velodyne")
LABEL_DIR = os.path.join(KITTI_DATA_PATH, "label_2")
IMAGE_DIR = os.path.join(KITTI_DATA_PATH, "image_2")
CALIB_DIR = os.path.join(KITTI_DATA_PATH, "calib")

for folder_path, folder_name in zip([CALIB_DIR, IMAGE_DIR], ['calib', 'image_2']):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📂 Created missing folder: {folder_name}")
    else:
        print(f"✅ Found folder: {folder_name}")

print(f"\n📁 REPO_ROOT: {REPO_ROOT}")
print(f"📂 KITTI_DATA_PATH: {KITTI_DATA_PATH}")

# ------------------------------#
# Utility Functions
# ------------------------------#
def rotate_eye_around_center(center, extent, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    radius = extent[0]
    eye_x = center[0] + radius * np.cos(angle_rad)
    eye_y = center[1] + radius * np.sin(angle_rad)
    eye_z = center[2] + extent[2] * 0.2
    return np.array([eye_x, eye_y, eye_z])


def compute_calibration_from_render(camera_fov_deg, center, eye, image_width, image_height):
    f = 0.5 * image_width / np.tan(np.deg2rad(camera_fov_deg) / 2)
    cx = image_width / 2
    cy = image_height / 2
    P2 = np.array([
        [f, 0, cx, 0],
        [0, f, cy, 0],
        [0, 0, 1, 0]
    ], dtype=np.float32)
    R0_rect = np.eye(3, dtype=np.float32)

    forward = center - eye
    forward /= np.linalg.norm(forward)
    tmp = np.array([0.0, 0.0, 1.0])
    right = np.cross(tmp, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    R = np.stack([right, up, forward], axis=0)
    t = np.zeros(3)

    Tr = np.eye(4, dtype=np.float32)
    Tr[:3, :3] = R
    Tr[:3, 3] = t

    return {
        "P2": P2,
        "R0_rect": R0_rect,
        "Tr_velo_to_cam": Tr
    }

def save_calib_kitti_format(path, calib_dict):
    def mat_to_line(tag, mat):
        return f"{tag}: " + " ".join([f"{v:.12e}" for v in mat.flatten()])
    
    dummy_P = np.zeros((3, 4), dtype=np.float32)
    dummy_Tr = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float32)

    with open(path, "w") as f:
        f.write(mat_to_line("P0", dummy_P) + "\n")
        f.write(mat_to_line("P1", dummy_P) + "\n")
        f.write(mat_to_line("P2", calib_dict["P2"]) + "\n")
        f.write(mat_to_line("P3", dummy_P) + "\n")
        f.write(mat_to_line("R0_rect", calib_dict["R0_rect"]) + "\n")
        f.write(mat_to_line("Tr_velo_to_cam", calib_dict["Tr_velo_to_cam"][:3]) + "\n")
        f.write(mat_to_line("Tr_imu_to_velo", dummy_Tr) + "\n")

    print(f"📝 Saved calibration file: {path}")

def render_pointcloud_to_image(points, output_png_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    depth = points[:, 0]
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-5)
    colors = plt.get_cmap("plasma")(depth_norm)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    w, h = 1280, 960
    renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
    renderer.scene.set_background([0, 0, 0, 1])

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.point_size = 5.0
    renderer.scene.add_geometry("pcd", pcd, mat)

    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.scene.set_sun_light(
        direction=[-1.0, -1.0, -1.0],
        color=[1.0, 1.0, 1.0],
        intensity=100000
    )

    bounds = pcd.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_extent()

    eye = rotate_eye_around_center(center, extent, angle_deg=180)
    up = np.array([0, 0, 1])
    fov = 60.0
    renderer.setup_camera(fov, center, eye, up)
    renderer.scene.camera.look_at(center, eye, up)

    print("\n📐 Calibration debug info:")
    print("🔸 eye (camera position):", eye)
    print("🔸 center (look-at point):", center)
    print("🔸 fov:", fov)
    print("🔸 image size:", w, "x", h, "\n")

    img = renderer.render_to_image()
    o3d.io.write_image(output_png_path, img)
    print(f"📸 Saved image: {output_png_path}")

    renderer.scene.clear_geometry()
    renderer = None

    return center, eye, w, h, fov

# ------------------------------#
# Main Loop
# ------------------------------#
def main():
    counterio=0
    
    velodyne_files = sorted([f for f in os.listdir(VELODYNE_DIR) if f.endswith('.bin')])[125:]
    print(f"🔍 Found {len(velodyne_files)} point cloud files\n")
    os.chdir(os.path.join(os.path.dirname(__file__)))  # Ensures script runs from the correct directory


    for filename in velodyne_files:
        start_time = time.time()
        filepath = os.path.join(VELODYNE_DIR, filename)
        points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)

        print(f"📦 Loaded {filename}: shape = {points.shape}")
        base_name = os.path.splitext(filename)[0]
        output_png_path = os.path.join(IMAGE_DIR, f"{base_name}.png")

        center, eye, w, h, fov = render_pointcloud_to_image(points, output_png_path)

        output_calib_path = os.path.join(CALIB_DIR, f"{base_name}.txt")
        calib = compute_calibration_from_render(fov, center, eye, w, h)
        save_calib_kitti_format(output_calib_path, calib)

        print(f"⏱️ Frame render time: {time.time() - start_time:.2f} sec\n")
        if(counterio==10):
            break
        # break  # Remove for full loop
        
# Run preprocessing
    print("🚀 Preprocessing KITTI dataset...")
    subprocess.run("python3 pre_process_kitti.py --data_root /home/user/deep_learning/Costume-Point-Pillars/datasets/INNOVIZ", shell=True, check=True)

# Run training from scratch
    # print("🧠 Starting training from scratch (no transfer learning)...")
    # subprocess.run("python3 train.py --data_root /lidar3d_detection_ws/training", shell=True, check=True)

    # counterio=0
    # for filename in velodyne_files:
    #     start_time = time.time()
    #     filepath = os.path.join(VELODYNE_DIR, filename)
    #     points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)

    #     print(f"📦 Loaded {filename}: shape = {points.shape}")
    #     base_name = os.path.splitext(filename)[0]
    #     output_png_path = os.path.join(IMAGE_DIR, f"{base_name}.png")

    #     center, eye, w, h, fov = render_pointcloud_to_image(points, output_png_path)

    #     output_calib_path = os.path.join(CALIB_DIR, f"{base_name}.txt")
    #     calib = compute_calibration_from_render(fov, center, eye, w, h)
    #     save_calib_kitti_format(output_calib_path, calib)

    #     print(f"⏱️ Frame render time: {time.time() - start_time:.2f} sec\n")
    #     if(counterio==10):
    #         break
        # break  # Remove for full loop

# ------------------------------#
# Entry Point
# ------------------------------#
if __name__ == "__main__":
    main()
















# def save_dummy_calibration(file_path):
#     def mat_to_str(m):
#         return " ".join(f"{v:.12e}" for v in m.flatten())

#     fx = fy = 721.5377
#     cx = 609.5593
#     cy = 172.8540

#     # Projection matrices (P0–P3)
#     P = np.array([[fx, 0, cx, 0],
#                   [0, fy, cy, 0],
#                   [0,  0,  1, 0]])

#     # Add slight variations to mimic real KITTI
#     P2 = P.copy()
#     P2[0, 3] = 44.85728
#     P2[1, 3] = 0.2163791
#     P2[2, 3] = 0.002745884

#     P3 = P.copy()
#     P3[0, 3] = -339.5242
#     P3[1, 3] = 2.199936
#     P3[2, 3] = 0.002729905

#     # Rotation matrix (R0_rect)
#     R0 = np.array([
#         [0.9999239,  0.00983776, -0.00744505],
#         [-0.0098698, 0.9999421,  -0.00427846],
#         [0.00740253, 0.00435161, 0.9999631]
#     ])

#     # Tr_velo_to_cam (LiDAR → Camera)
#     Tr_velo_to_cam = np.array([
#         [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
#         [1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
#         [9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01]
#     ])

#     # Tr_imu_to_velo
#     Tr_imu_to_velo = np.array([
#         [9.999976e-01,  7.553071e-04, -2.035826e-03, -8.086759e-01],
#         [-7.854027e-04, 9.998898e-01, -1.482298e-02,  3.195559e-01],
#         [2.024406e-03,  1.482454e-02,  9.998881e-01, -7.997231e-01]
#     ])

#     with open(file_path, "w") as f:
#         f.write(f"P0: {mat_to_str(P)}\n")
#         f.write(f"P1: {mat_to_str(P)}\n")
#         f.write(f"P2: {mat_to_str(P2)}\n")
#         f.write(f"P3: {mat_to_str(P3)}\n")
#         f.write(f"R0_rect: {mat_to_str(R0)}\n")
#         f.write(f"Tr_velo_to_cam: {mat_to_str(Tr_velo_to_cam)}\n")
#         f.write(f"Tr_imu_to_velo: {mat_to_str(Tr_imu_to_velo)}\n")