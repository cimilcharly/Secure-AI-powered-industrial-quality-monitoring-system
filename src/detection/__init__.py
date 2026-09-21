from .yolo_detector import (
    YOLOv8DefectDetector,
    DEFECT_CLASSES,
    CLASS_COLORS,
    detect_defects
)

__all__ = [
    "YOLOv8DefectDetector",
    "DEFECT_CLASSES",
    "CLASS_COLORS",
    "detect_defects"
]
