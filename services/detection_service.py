"""Detection workflow service coordinating validation, preprocessing, inference, and history."""

from typing import Any, Optional
from PIL import Image

from config.settings import DEFAULT_CONFIDENCE, DEFAULT_IMAGE_SIZE, DEFAULT_IOU
from core.detector import DetectionResult, detect_candies
from core.image_processor import preprocess_image, validate_image_upload
from services.history_service import HistoryService
from utils.validators import validate_thresholds


class DetectionService:
    """Coordinates end-to-end detection pipelines for uploaded photos and camera snapshots."""

    @classmethod
    def process_image(
        cls,
        image_input: Any,
        filename: str = "candy_scan.jpg",
        confidence: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU,
        imgsz: int = DEFAULT_IMAGE_SIZE,
        log_to_history: bool = True,
    ) -> DetectionResult:
        """
        Execute full detection lifecycle:
        1. Validates thresholds
        2. Preprocesses image into standardized RGB PIL Image
        3. Executes YOLOv8 inference
        4. Logs scan to session history
        5. Returns structured DetectionResult
        """
        # Validate parameters
        thresh_ok, thresh_err = validate_thresholds(confidence, iou)
        if not thresh_ok:
            raise ValueError(thresh_err)

        # Preprocess image
        pil_image = preprocess_image(image_input)

        # Execute inference
        result = detect_candies(
            image=pil_image,
            confidence=confidence,
            iou=iou,
            imgsz=imgsz,
        )

        # Log to session history
        if log_to_history:
            HistoryService.add_record(
                filename=filename,
                result=result,
                confidence=confidence,
                iou=iou,
            )

        return result

    @classmethod
    def validate_upload(cls, uploaded_file: Any) -> Optional[str]:
        """Validate an uploaded file and return an error message if invalid."""
        is_valid, msg = validate_image_upload(uploaded_file)
        if not is_valid:
            return msg
        return None

