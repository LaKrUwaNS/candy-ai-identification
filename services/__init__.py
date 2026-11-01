"""Services package coordinating application workflows."""

from services.camera_service import CameraService, FPSCalculator
from services.detection_service import DetectionService
from services.history_service import HistoryService
from services.statistics_service import StatisticsService

__all__ = [
    "DetectionService",
    "HistoryService",
    "StatisticsService",
    "CameraService",
    "FPSCalculator",
]

