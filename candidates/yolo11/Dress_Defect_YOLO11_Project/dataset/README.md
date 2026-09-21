# Dataset folder

Place your dataset here in this structure:

dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml

For every image, the corresponding YOLO label `.txt` file goes in the
matching labels folder.

Example:
images/train/dress001.jpg
labels/train/dress001.txt
