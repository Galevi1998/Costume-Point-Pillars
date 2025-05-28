import os
from collections import Counter
from pathlib import Path

# ✅ הנתיב לתיקיית ה-GT
GT_DATABASE_DIR = Path("/lidar3d_detection_ws/training/kitti_gt_database")

if not GT_DATABASE_DIR.exists():
    raise FileNotFoundError(f"❌ GT folder not found at: {GT_DATABASE_DIR}")

# 🔍 קריאה של כל קבצי ה-.bin
all_files = [f for f in os.listdir(GT_DATABASE_DIR) if f.endswith(".bin")]

if not all_files:
    print("⚠️ No .bin files found in GT database folder.")
else:
    # 📦 חילוץ שמות המחלקות (Car, Pedestrian וכו')
    class_names = []
    for fname in all_files:
        parts = fname.split("_")
        if len(parts) >= 3:
            class_names.append(parts[1])
        else:
            print(f"⚠️ Skipping malformed file name: {fname}")

    # 📊 ספירה
    class_counts = Counter(class_names)

    print("\n📊 Class distribution in kitti_gt_database:")
    print("-" * 35)
    total = 0
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"{cls:>12}: {count} samples")
        total += count
    print("-" * 35)
    print(f"{'Total':>12}: {total} samples\n")


# import numpy as np
# from pointpillars.utils import read_points, read_label, read_calib
# from pointpillars.utils import points_in_bboxes_v2

# id = "00026"  # לדוגמה קובץ בעייתי
# class_name = "Car"
# db_idx = 0

# lidar_path = f"/lidar3d_detection_ws/training/training/velodyne/{id}.bin"
# label_path = f"/lidar3d_detection_ws/training/training/label_2/{id}.txt"
# calib_path = f"/lidar3d_detection_ws/training/training/calib/{id}.txt"

# points = read_points(lidar_path)
# labels = read_label(label_path)
# calib = read_calib(calib_path)

# # ננסה לראות אילו תיבות מתאימות למחלקה הזו
# name_mask = labels["name"] == class_name
# dims = labels["dimensions"][name_mask]
# locs = labels["location"][name_mask]
# rots = labels["rotation_y"][name_mask]
# names = labels["name"][name_mask]

# indices, _, valid, _, _ = points_in_bboxes_v2(
#     points,
#     r0_rect=calib["R0_rect"],
#     tr_velo_to_cam=calib["Tr_velo_to_cam"],
#     dimensions=dims.astype(np.float32),
#     location=locs.astype(np.float32),
#     rotation_y=rots.astype(np.float32),
#     name=names
# )

# print(f"Found {valid} valid boxes containing points.")

# import os
# import numpy as np
# from collections import Counter

# GT_DATABASE_DIR = "/lidar3d_detection_ws/training/kitti_gt_database"

# if not os.path.exists(GT_DATABASE_DIR):
#     print(f"Directory does not exist: {GT_DATABASE_DIR}")
#     exit(1)

# all_files = os.listdir(GT_DATABASE_DIR)
# class_names = []
# bad_files = []

# for fname in all_files:
#     if fname.endswith(".bin"):
#         parts = fname.split("_")
#         if len(parts) < 3:
#             print(f"⚠️ Irregular file name: {fname}")
#             continue
#         class_names.append(parts[1])
#         # תקינות תוכן הקובץ
#         path = os.path.join(GT_DATABASE_DIR, fname)
#         try:
#             points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
#             if points.shape[0] == 0:
#                 print(f"⚠️ Empty point cloud: {fname}")
#         except Exception as e:
#             print(f"❌ Failed to read {fname}: {e}")
#             bad_files.append(fname)

# # הדפסת סטטיסטיקה
# class_counts = Counter(class_names)
# print("\nClass distribution in kitti_gt_database:")
# print("-----------------------------------")
# for cls, count in sorted(class_counts.items()):
#     print(f"{cls:>12}: {count} samples")
# print("-----------------------------------")
# print(f"{'Total':>12}: {sum(class_counts.values())} samples")


import os
import numpy as np

GT_DATABASE_DIR = "/lidar3d_detection_ws/training/kitti_gt_database"

valid_files = []
empty_files = []

for file in os.listdir(GT_DATABASE_DIR):
    if file.endswith(".bin"):
        path = os.path.join(GT_DATABASE_DIR, file)
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        if points.shape[0] > 0:
            valid_files.append(file)
        else:
            empty_files.append(file)

print("✅ Files with useful points:", len(valid_files))
print("❌ Empty files (not useful for db_sampling):", len(empty_files))
print("📦 Total .bin files:", len(valid_files) + len(empty_files))
print("\n📝 Examples of valid files:", valid_files[:5])
print("🗑️  Examples of empty files:", empty_files[:5])


import os
import pickle
import numpy as np
from collections import defaultdict

# נתיב לקובץ infos שנוצר ע"י הסקריפט pre_process_kitti
INFO_PATH = "/lidar3d_detection_ws/training/kitti_infos_train.pkl"

# טוען את המידע
with open(INFO_PATH, 'rb') as f:
    infos = pickle.load(f)

# נבנה מילון שיאגור את המידות לכל מחלקה
dims_per_class = defaultdict(list)

for frame_id, frame_info in infos.items():
    annos = frame_info.get("annos", {})
    names = annos.get("name", [])
    dims = annos.get("dimensions", [])
    
    for name, dim in zip(names, dims):
        dims_per_class[name].append(dim)  # dim in lhw order (KITTI)

# הדפסה של הממוצעים
print("📏 Mean dimensions per class (Length, Height, Width):")
print("---------------------------------------------------")
for cls, dims in dims_per_class.items():
    dims_np = np.array(dims)
    mean_dims = dims_np.mean(axis=0)
    print(f"{cls:>12}: L={mean_dims[0]:.2f}, H={mean_dims[1]:.2f}, W={mean_dims[2]:.2f}  (n={len(dims)})")




import numpy as np
import os

def read_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    P2 = np.array([float(x) for x in lines[2].split()[1:]]).reshape(3, 4)
    R0_rect = np.array([float(x) for x in lines[4].split()[1:]]).reshape(3, 3)
    Tr_velo_to_cam = np.array([float(x) for x in lines[5].split()[1:]]).reshape(3, 4)
    return {'P2': P2, 'R0_rect': R0_rect, 'Tr_velo_to_cam': Tr_velo_to_cam}

def read_label(label_path):
    annos = {'dimensions': [], 'location': [], 'rotation_y': []}
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split(' ')
        h, w, l = map(float, parts[8:11])
        loc = list(map(float, parts[11:14]))
        ry = float(parts[14])
        annos['dimensions'].append([h, w, l])
        annos['location'].append(loc)
        annos['rotation_y'].append(ry)
    annos['dimensions'] = np.array(annos['dimensions'])
    annos['location'] = np.array(annos['location'])
    annos['rotation_y'] = np.array(annos['rotation_y'])
    return annos

def cam_to_lidar_box3d(location, dimensions, rotation_y, Tr_velo_to_cam, R0_rect):
    N = location.shape[0]
    loc_rect = (R0_rect @ location.T).T
    loc_rect_homo = np.hstack((loc_rect, np.ones((N, 1))))
    Tr = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    Tr_inv = np.linalg.inv(Tr)
    loc_lidar = (Tr_inv @ loc_rect_homo.T).T[:, :3]
    h, w, l = dimensions[:, 0], dimensions[:, 1], dimensions[:, 2]
    yaw_lidar = -rotation_y - np.pi / 2
    boxes_lidar = np.stack([loc_lidar[:, 0], loc_lidar[:, 1], loc_lidar[:, 2], w, l, h, yaw_lidar], axis=1)
    return boxes_lidar

def compute_anchor_range_from_boxes(boxes_lidar):
    x_min, y_min, z_min = boxes_lidar[:, :3].min(0)
    x_max, y_max, z_max = boxes_lidar[:, :3].max(0)
    return [round(float(x_min), 2), round(float(y_min), 2), round(float(z_min), 2),
            round(float(x_max), 2), round(float(y_max), 2), round(float(z_max), 2)]

# 🔁 MODIFY THESE PATHS TO MATCH YOUR FILES:
# calib_path = '/path/to/calib_train_2001/000003.txt'
# label_path = '/path/to/labels_train_first_2001/000003.txt'
id = "00022"  # לדוגמה קובץ בעייתי
class_name = "Car"
db_idx = 0

lidar_path = f"/lidar3d_detection_ws/training/training/velodyne/{id}.bin"
label_path = f"/lidar3d_detection_ws/training/training/label_2/{id}.txt"
calib_path = f"/lidar3d_detection_ws/training/training/calib/{id}.txt"

# ✅ RUN:
calib = read_calib(calib_path)
annos = read_label(label_path)
boxes_lidar = cam_to_lidar_box3d(annos['location'], annos['dimensions'], annos['rotation_y'],
                                  calib['Tr_velo_to_cam'], calib['R0_rect'])
anchor_range = compute_anchor_range_from_boxes(boxes_lidar)
print("🔧 Suggested Anchor Range:", anchor_range)
