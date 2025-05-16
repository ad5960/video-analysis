import torch
from ultralytics import YOLO
import logging
import os
import time
from typing import Union, List, Dict # For Python 3.9 compatibility

logger = logging.getLogger(__name__)

# --- Confidence Threshold ---
# You can also make this configurable via environment variable
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
logger.info(f"Using confidence threshold: {CONFIDENCE_THRESHOLD}")
# --- End Confidence Threshold ---

class ModelHandler:
    def __init__(self, model_path: str):
        self.device = self._get_device()
        self.model = self._load_model(model_path)

    def _get_device(self) -> torch.device:
        """Sets the device to MPS if available, otherwise CPU."""
        if torch.backends.mps.is_available():
            logger.info("MPS device found. Using MPS.")
            return torch.device("mps")
        # Optional: Check for CUDA as well if planning for NVIDIA deployment
        # elif torch.cuda.is_available():
        #     logger.info("CUDA device found. Using CUDA.")
        #     return torch.device("cuda")
        else:
            logger.info("No GPU accelerator found. Using CPU.")
            return torch.device("cpu")

    def _load_model(self, model_path: str) -> Union[YOLO, None]:
        """Loads the YOLOv8 model onto the determined device."""
        if not os.path.exists(model_path):
             logger.error(f"Model file not found at: {model_path}")
             return None
        try:
            logger.info(f"Loading model from {model_path} onto {self.device}...")
            start_time = time.time()
            # Load the model using Ultralytics YOLO class
            model = YOLO(model_path)
            logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds.")
            # Perform a dummy inference to potentially warm up MPS/GPU
            try:
                 logger.info("Performing dummy inference for warmup...")
                 _ = model.predict(torch.zeros(1, 3, 640, 640).to(self.device), verbose=False)
                 logger.info("Warmup inference complete.")
            except Exception as warmup_e:
                 logger.warning(f"Warmup inference failed: {warmup_e}")

            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True) # Log traceback
            return None

    def run_inference(self, image) -> Union[List, None]:
        """Runs YOLOv8 inference on the preprocessed image."""
        if self.model is None:
             logger.error("Model is not loaded. Cannot run inference.")
             return None
        try:
            # Perform inference using the YOLO object's predict method
            # It handles preprocessing and uses the best available device
            results = self.model.predict(source=image, device=self.device, verbose=False)
            return results
        except Exception as e:
            logger.error(f"Error during model inference: {e}", exc_info=True) # Log traceback
            return None

    def format_results(self,
                       results: Union[List, None],
                       frame_id: str,
                       wall_timestamp: float,
                       video_timestamp_ms: Union[float, None],
                       camera_id: str) -> Union[Dict, None]:
        """Formats the raw YOLO results into the desired JSON structure, applying confidence threshold."""
        if not results or len(results) == 0:
            # Return structure with empty objects list but include timestamps
            return {
                "frame_id": frame_id,
                "wall_timestamp": wall_timestamp,
                "video_timestamp_ms": video_timestamp_ms,
                "camera_id": camera_id,
                "objects": []
            }

        # Process the first result object (assuming single image inference)
        result = results[0]
        detected_objects = []

        if result.boxes is not None:
            for box in result.boxes:
                try:
                    confidence = float(box.conf)
                    # --- >>> ADD CONFIDENCE THRESHOLD CHECK HERE <<< ---
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue # Skip this detection if below threshold
                    # --- >>> END CONFIDENCE THRESHOLD CHECK <<< ---

                    class_id = int(box.cls)
                    class_name = self.model.names.get(class_id, f"unknown_class_{class_id}")
                    bbox_coords = box.xyxy[0].cpu().numpy().tolist() # [xmin, ymin, xmax, ymax]

                    obj_data = {
                        "object_id": None, # To be assigned by post-processing
                        "class": class_name,
                        "confidence": confidence, # Store the original confidence
                        "bbox": bbox_coords,
                    }
                    detected_objects.append(obj_data)
                except Exception as e:
                    logger.error(f"Frame {frame_id}: Error processing a detection box: {e}", exc_info=True)
                    continue # Skip this box

        output_json = {
            "frame_id": frame_id,
            "wall_timestamp": wall_timestamp,
            "video_timestamp_ms": video_timestamp_ms,
            "camera_id": camera_id,
            "objects": detected_objects # List will only contain objects above threshold
        }
        return output_json
