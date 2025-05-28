import os
from pathlib import Path
from random import shuffle

# Define the frame groups based on user's scene information
scene_1 = list(range(20, 89))     # Scene A
scene_2 = list(range(89, 120))    # Scene B
scene_3 = list(range(0, 20)) + list(range(120, 243))  # Scene C

# Shuffle and split
shuffle(scene_1)
shuffle(scene_2)
shuffle(scene_3)

# For overfitting: use Scene C mostly for training, and take some from Scene A and B
train_ids = scene_3 + scene_1[:40] + scene_2[:20]  # train on most of Scene C and some from A/B
val_ids = scene_1[40:] + scene_2[20:]              # validation from rest of A/B
trainval_ids = train_ids + val_ids
test_ids = list(range(0, 243))                     # test includes all

# Convert to padded strings
def format_ids(ids):
    return [f"{i:05d}" for i in sorted(ids)]

files = {
    "train.txt": format_ids(train_ids),
    "val.txt": format_ids(val_ids),
    "trainval.txt": format_ids(trainval_ids),
    "test.txt": format_ids(test_ids),
}

# Save files
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
print(REPO_ROOT)
base_path = Path(f"{REPO_ROOT}/pointpillars/dataset/ImageSets")
os.makedirs(base_path, exist_ok=True)

for name, lines in files.items():
    with open(base_path / name, "w") as f:
        f.write("\n".join(lines))


