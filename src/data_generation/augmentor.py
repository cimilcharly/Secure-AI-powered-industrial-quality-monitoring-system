"""
Data Augmentation & Synthetic Expansion Pipeline (Module 2).
Provides classical augmentation with Albumentations (flips, rotations, color jitter, shift-scale-rotate)
and conditional synthetic defect pattern blending per confirmed category.
"""

import os
import random
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


class FabricAugmentor:
    """Industrial augmentation pipeline for fabric defect images and bounding boxes."""

    def __init__(self, p: float = 0.85):
        self.p = p
        if ALBUMENTATIONS_AVAILABLE:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=20, p=0.6, border_mode=cv2.BORDER_REFLECT),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.OneOf([
                    A.MotionBlur(blur_limit=5, p=0.5),
                    A.GaussianBlur(blur_limit=5, p=0.5),
                ], p=0.3),
            ], bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids']))
        else:
            self.transform = None

    def augment_sample(
        self,
        image_bgr: np.ndarray,
        bboxes: List[List[float]],
        category_ids: List[int]
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Applies transformations to image and normalized bounding boxes [x1, y1, x2, y2].
        """
        if self.transform is not None and len(bboxes) > 0:
            try:
                transformed = self.transform(
                    image=image_bgr,
                    bboxes=bboxes,
                    category_ids=category_ids
                )
                return transformed['image'], transformed['bboxes'], transformed['category_ids']
            except Exception:
                pass

        # Fallback OpenCV operations
        aug_img = image_bgr.copy()
        if random.random() > 0.5:
            aug_img = cv2.flip(aug_img, 1)  # horizontal flip
        if random.random() > 0.5:
            # Color jitter
            val = random.uniform(0.8, 1.2)
            hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val, 0, 255)
            aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return aug_img, bboxes, category_ids

    def generate_synthetic_defect(
        self,
        nominal_bgr: np.ndarray,
        category: str = "Damage"
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Synthetically injects realistic defect pattern into nominal fabric.
        Returns: (augmented_image, [x1, y1, x2, y2] bounding box)
        """
        img = nominal_bgr.copy()
        h, w = img.shape[:2]

        cx = random.randint(int(w * 0.25), int(w * 0.75))
        cy = random.randint(int(h * 0.25), int(h * 0.75))
        size = random.randint(int(min(w, h) * 0.08), int(min(w, h) * 0.18))

        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(w - 1, cx + size // 2)
        y2 = min(h - 1, cy + size // 2)

        roi = img[y1:y2, x1:x2]

        if category == "Damage":
            # Dark tear or frayed hole
            cv2.ellipse(img, (cx, cy), (size // 2, size // 4), random.randint(0, 180), 0, 360, (20, 20, 25), -1)
            cv2.circle(img, (cx, cy), size // 6, (10, 10, 15), -1)
        elif category == "Color":
            # Dye stain / bleach spot
            color = (random.randint(180, 255), random.randint(50, 120), random.randint(40, 100))
            overlay = img.copy()
            cv2.circle(overlay, (cx, cy), size // 2, color, -1)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        elif category == "Button":
            # Misaligned / cracked button simulation
            cv2.circle(img, (cx, cy), size // 3, (220, 220, 230), -1)
            cv2.circle(img, (cx, cy), size // 3, (120, 120, 130), 2)
            cv2.line(img, (cx - size // 4, cy - size // 4), (cx + size // 4, cy + size // 4), (40, 40, 40), 2)
        elif category == "Stitch":
            # Loose thread / skipped stitch line
            pts = np.array([
                [cx - size // 2, cy],
                [cx - size // 6, cy - size // 4],
                [cx + size // 6, cy + size // 3],
                [cx + size // 2, cy - size // 5]
            ], np.int32)
            cv2.polylines(img, [pts], False, (240, 240, 50), 3)

        return img, [x1, y1, x2, y2]


_default_augmentor = FabricAugmentor()


def augment_fabric_sample(image_bgr: np.ndarray, bboxes: list, category_ids: list):
    """Helper function for sample augmentation."""
    return _default_augmentor.augment_sample(image_bgr, bboxes, category_ids)


def generate_synthetic_variations(nominal_bgr: np.ndarray, category: str = "Damage"):
    """Helper function to create synthetic variation."""
    return _default_augmentor.generate_synthetic_defect(nominal_bgr, category)
