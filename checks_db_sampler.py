import os
import pickle
import random
import pandas as pd
# import ace_tools as tools

PKL_FILES = [
    "kitti_infos_train.pkl",
    "kitti_infos_val.pkl",
    "kitti_infos_trainval.pkl",
    "kitti_infos_test.pkl",
    "kitti_dbinfos_train.pkl",
]

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def analyze_infos(file_path, infos):
    print(f"\n📁 {file_path}")
    print(f"▪ Total samples: {len(infos)}")

    if isinstance(infos, dict):
        sample_key = next(iter(infos))
        sample = infos[sample_key]
    elif isinstance(infos, list):
        sample = infos[0] if len(infos) > 0 else None
    else:
        sample = None

    if sample:
        print(f"▪ Sample keys: {list(sample.keys())}")

def analyze_dbinfos(file_path, dbinfos):
    print(f"\n📁 {file_path}")
    print(f"▪ Classes in DB: {list(dbinfos.keys())}")
    total = 0
    for cls in dbinfos:
        count = len(dbinfos[cls])
        total += count
        print(f"  ▸ {cls}: {count} samples")
    print(f"▪ Total DB samples: {total}")



def main(data_dir):
    for fname in PKL_FILES:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"\n❌ Missing: {fname}")
            continue

        try:
            data = load_pickle(path)
        except Exception as e:
            print(f"\n❌ Failed to load {fname}: {e}")
            continue

        if fname.startswith("kitti_dbinfos"):
            analyze_dbinfos(fname, data)
        else:
            analyze_infos(fname, data)

        # Define path to the dbinfo file
    dbinfo_path = "/lidar3d_detection_ws/training/kitti_dbinfos_train.pkl"

    # Load dbinfo file
    with open(dbinfo_path, 'rb') as f:
        dbinfo = pickle.load(f)

    # Inspect each class in the database
    summary = {}
    for cls_name, samples in dbinfo.items():
        random_sample = random.choice(samples)
        summary[cls_name] = {
            'total_samples': len(samples),
            'example_box': random_sample['box3d_lidar'] if samples else None,
            'example_path': random_sample['path'] if samples else None,
            'example_points': random_sample['num_points_in_gt'] if samples else None,
            'example_difficulty': random_sample['difficulty'] if samples else None
        }
    df = pd.DataFrame(summary).T
    print("\n📊 Summary of DB Samples:")
    print(df)
    


if __name__ == "__main__":
    data_root = "/lidar3d_detection_ws/training"  # עדכן לפי המיקום אצלך
    main(data_root)
