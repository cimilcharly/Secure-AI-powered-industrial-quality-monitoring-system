"""
Image Preprocessing Pipeline for Fabric Defect Inspection.
Shared across training, inference, and live dashboard feeds.
Includes standardization, CLAHE contrast enhancement, noise reduction, and normalization.
"""

import io
from typing import Tuple, Union, Optional
import cv2
import numpy as np
from PIL import Image


class FabricPreprocessor:
    """Industrial image preprocessor for fabric defect and anomaly inspection."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        apply_clahe: bool = True,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        apply_denoise: bool = True,
        denoise_kernel: int = 3
    ):
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.apply_denoise = apply_denoise
        self.denoise_kernel = denoise_kernel

        # Setup OpenCV CLAHE instance
        if self.apply_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size
            )
        else:
            self.clahe = None

    def enhance_contrast(self, image_bgr: np.ndarray) -> np.ndarray:
        """Applies CLAHE on the L-channel of LAB color space to avoid color distortion."""
        if not self.apply_clahe or self.clahe is None:
            return image_bgr

        # Convert BGR to LAB
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L-channel
        cl = self.clahe.apply(l)

        # Merge back and convert to BGR
        limg = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

    def reduce_noise(self, image_bgr: np.ndarray) -> np.ndarray:
        """Applies gentle bilateral or Gaussian filtering to suppress sensor grain while keeping defect edges sharp."""
        if not self.apply_denoise:
            return image_bgr
        # Bilateral filter preserves sharp edges (scratches, holes, threads)
        denoised = cv2.bilateralFilter(image_bgr, d=5, sigmaColor=35, sigmaSpace=35)
        return denoised

    def resize_and_pad(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Letterbox resize with aspect ratio preservation and padding.
        Returns: (padded_image, scale_factor, (pad_w, pad_h))
        """
        target_w, target_h = self.target_size
        h, w = image_bgr.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create target canvas with neutral gray padding (114 for YOLO compatibility)
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        return canvas, scale, (pad_x, pad_y)

    def process(
        self,
        image_input: Union[np.ndarray, Image.Image, bytes, str]
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Full standardized pipeline execution.
        Returns:
            - processed_bgr (np.ndarray): resized, enhanced, uint8 BGR image
            - normalized_float (np.ndarray): (3, H, W) float32 normalized [0.0, 1.0] tensor
            - metadata (dict): original dimensions, scale, padding info
        """
        # 1. Standardize input to BGR numpy array
        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise FileNotFoundError(f"Could not read image from path: {image_input}")
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input.copy()
            if len(img_bgr.shape) == 2:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        orig_h, orig_w = img_bgr.shape[:2]

        # 2. Contrast enhancement
        enhanced = self.enhance_contrast(img_bgr)

        # 3. Noise reduction
        denoised = self.reduce_noise(enhanced)

        # 4. Standardized resize & letterbox padding
        processed_bgr, scale, (pad_x, pad_y) = self.resize_and_pad(denoised)

        # 5. Tensor normalization: BGR -> RGB -> [0, 1] -> CHW
        rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        normalized_float = rgb.astype(np.float32) / 255.0
        normalized_float = np.transpose(normalized_float, (2, 0, 1))

        metadata = {
            "original_width": orig_w,
            "original_height": orig_h,
            "target_size": self.target_size,
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y
        }

        return processed_bgr, normalized_float, metadata


# Global singleton instance for quick usage
_default_preprocessor = FabricPreprocessor()


def preprocess_image(image_input: Union[np.ndarray, Image.Image, bytes, str]):
    """Helper function to run preprocessing using the default preprocessor instance."""
    return _default_preprocessor.process(image_input)


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode raw bytes into a BGR OpenCV image."""
    nparr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def image_to_bytes(image_bgr: np.ndarray, format: str = ".jpg", quality: int = 90) -> bytes:
    """Encode OpenCV BGR image into bytes."""
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if format.lower() in [".jpg", ".jpeg"] else []
    success, buffer = cv2.imencode(format, image_bgr, params)
    if not success:
        raise ValueError("Failed to encode image to bytes")
    return buffer.tobytes()
