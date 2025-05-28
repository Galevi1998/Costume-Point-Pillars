import numpy as np
import os

def rotate_eye_around_center(center, extent, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    radius = extent[0]
    eye_x = center[0] + radius * np.cos(angle_rad)
    eye_y = center[1] + radius * np.sin(angle_rad)
    eye_z = center[2] + extent[2] * 0.2
    return np.array([eye_x, eye_y, eye_z])


def compute_calibration_from_render(camera_fov_deg, center, eye, image_width, image_height):
    # 1. Compute intrinsic matrix (P2)
    f = 0.5 * image_width / np.tan(np.deg2rad(camera_fov_deg) / 2)
    cx = image_width / 2
    cy = image_height / 2
    P2 = np.array([
        [f, 0, cx, 0],
        [0, f, cy, 0],
        [0, 0, 1, 0]
    ])

    # 2. Compute rectification matrix R0_rect (identity)
    R0_rect = np.eye(3)

    # 3. Compute extrinsic matrix Tr_velo_to_cam (rotation + translation)
    forward = center - eye
    forward /= np.linalg.norm(forward)

    tmp = np.array([0.0, 0.0, 1.0])
    right = np.cross(tmp, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    R = np.stack([right, up, forward], axis=0)
    t = -R @ eye

    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :3] = R
    Tr_velo_to_cam[:3, 3] = t

    print("📐 Calibration debug info:")
    print("🔸 eye (camera position):", eye)
    print("🔸 center (look-at point):", center)
    print("🔸 fov (field of view):", camera_fov_deg)
    print("🔸 image width, height:", image_width, image_height)
    print("🔸 Intrinsic matrix P2:\n", P2)
    print("🔸 R0_rect:\n", R0_rect)
    print("🔸 Tr_velo_to_cam:\n", Tr_velo_to_cam[:3])

    return {
        "P2": P2,
        "R0_rect": R0_rect,
        "Tr_velo_to_cam": Tr_velo_to_cam
    }


def save_calib_kitti_format(path, calib_dict):
    def mat_to_line(tag, mat):
        flat = mat.flatten()
        return f"{tag}: " + " ".join([f"{x:.12e}" for x in flat])

    with open(path, 'w') as f:
        f.write(mat_to_line("P2", calib_dict["P2"]) + "\n")
        f.write(mat_to_line("R0_rect", calib_dict["R0_rect"]) + "\n")
        f.write(mat_to_line("Tr_velo_to_cam", calib_dict["Tr_velo_to_cam"][:3, :]) + "\n")

    print(f"✅ Saved KITTI-format calib to: {path}")


if __name__ == "__main__":
    # Example values — replace with your actual data
    center = np.array([15.718, 0.0, 1.446])  # center of bounding box
    extent = np.array([28.6, 0.0, 2.3])      # approximate scene size
    eye = rotate_eye_around_center(center, extent, angle_deg=180)

    fov = 60.0
    w, h = 1280, 960

    calib = compute_calibration_from_render(fov, center, eye, w, h)

    output_path = "calib.txt"
    save_calib_kitti_format(output_path, calib)
