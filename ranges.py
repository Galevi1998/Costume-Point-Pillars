# ranges.py

import pickle
import numpy as np
from collections import defaultdict

def compute_anchor_ranges(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    name_to_locations = defaultdict(list)

    for sample_id, info in data.items():
        annos = info.get('annos', {})
        names = annos.get('name', [])
        locs = annos.get('location', [])

        for name, loc in zip(names, locs):
            name_to_locations[name].append(loc)

    print("🔍 Total object classes:", list(name_to_locations.keys()))
    print()

    for name, locs in name_to_locations.items():
        locs_np = np.array(locs)
        x_min, y_min, z_min = locs_np.min(axis=0)
        x_max, y_max, z_max = locs_np.max(axis=0)

        # Add some padding to avoid edge cases being missed
        padding = 0.5
        range_with_padding = [
            round(x_min - padding, 2), 
            round(y_min - padding, 2), 
            round(z_min - padding, 2), 
            round(x_max + padding, 2), 
            round(y_max + padding, 2), 
            round(z_max + padding, 2),
        ]

        print(f"🔧 Class: {name}")
        print(f"Anchor Range: {range_with_padding}")
        print()

if __name__ == '__main__':
    # Set this path to where your actual pickle file is
    PKL_PATH = '/lidar3d_detection_ws/training/kitti_infos_trainval.pkl'
    compute_anchor_ranges(PKL_PATH)
