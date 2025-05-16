import base64
import cv2
import numpy as np
import logging
from typing import Union

logger = logging.getLogger(__name__)

def decode_image_from_base64(base64_string: str) -> Union[np.ndarray, None]:
    """Decodes a base64 encoded image string into a NumPy array."""
    # ... (rest of the function remains the same)
    try:
        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to decode image from byte array.")
            return None
        return img
    except (base64.binascii.Error, ValueError, TypeError) as e:
        logger.error(f"Error decoding base64 string: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during image decoding: {e}")
        return None

def preprocess_for_yolo(image: np.ndarray, target_size: tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Prepares an image for YOLOv8 inference.
    Specifically: Resizes and potentially converts color space if needed
    (YOLOv8 often handles normalization internally when using its predict method,
    and expects BGR numpy arrays by default).
    Check Ultralytics documentation for specifics if not using model.predict().
    """
    # YOLOv8's predict method typically handles resizing and normalization.
    # If you were manually creating the input tensor, you would:
    # 1. Resize (maintaining aspect ratio with padding if needed)
    # 2. Convert BGR to RGB (if model expects RGB)
    # 3. Convert to float32
    # 4. Normalize pixel values (e.g., divide by 255.0)
    # 5. Add batch dimension and permute channels (e.g., HWC to CHW)

    # For simplicity here, we assume the model handler's predict call handles this.
    # We just ensure the image is valid.
    # If specific resizing is needed *before* model.predict, add it here.
    # Example: Resize keeping aspect ratio might be needed if not using model.predict()
    # img_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # Ensure image is in BGR format as OpenCV reads it
    return image