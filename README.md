# Custom PointPillars: Modified 3D Object Detection from Point Clouds

This repository is a **customized and modified version** of the original [PointPillars implementation by zhulf0804](https://github.com/zhulf0804/PointPillars), focused on fine-tuning and adapting the model for different performance goals and dataset variations.

## 🔄 What's Different in This Version

- Modified model architecture and training configuration
- Different data augmentation techniques
- Adjusted evaluation metrics and visualizations
- Different results compared to the original implementation (see below)

> 📌 **Note**: This code is **not officially maintained** by the original authors. It is a derivative work maintained by **[@Galevi1998](https://github.com/Galevi1998)**.

## 📊 My Results

Compared to the original repo, this implementation yielded **different performance** metrics on the KITTI validation set due to changes in training, preprocessing, and possibly hardware.

(Add your own mAP or results table here if you'd like.)

## 🧠 Credits

This work is based on the original open-source implementation:  
**👉 [zhulf0804/PointPillars](https://github.com/zhulf0804/PointPillars)**

Thanks to the original authors: `zhulifa`, `raphaelsulzer`, `zhulf0804`, and all contributors.

Also thanks to the broader 3D detection ecosystem:  
[mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [mmdetection](https://github.com/open-mmlab/mmdetection), and [mmcv](https://github.com/open-mmlab/mmcv).

## 🚀 Installation

```bash
cd PointPillars/
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install .
