"""Centralized YOLOv8 model loading, device detection, and validation."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
from ultralytics import YOLO

from config.settings import CANDY_CLASSES, MODEL_PATH
from utils.file_handler import check_model_exists, resolve_model_path

logger = logging.getLogger(__name__)

# Module-level singleton cache for non-Streamlit environments (e.g. tests, CLI)
_MODEL_CACHE: Optional[YOLO] = None


def get_compute_device() -> str:
    """Detect available compute device (CUDA GPU or CPU fallback)."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device_info() -> Dict[str, str]:
    """Retrieve detailed hardware device descriptions."""
    device = get_compute_device()
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        capability = str(torch.cuda.get_device_capability(0))
        return {
            "device": "CUDA (GPU)",
            "device_name": name,
            "capability": capability,
            "cuda_version": torch.version.cuda or "Unknown",
        }
    return {
        "device": "CPU",
        "device_name": "Host Processor Engine",
        "capability": "N/A",
        "cuda_version": "N/A",
    }


def load_raw_model(weights_path: Optional[Path] = None) -> YOLO:
    """
    Load YOLO model directly from specified or resolved weights file.
    Does not depend on Streamlit decorators.
    """
    target_path = weights_path or resolve_model_path()
    if not target_path.is_file():
        exists, err_msg = check_model_exists()
        raise FileNotFoundError(err_msg)

    logger.info("Loading YOLO weights from %s", target_path)
    model = YOLO(str(target_path))
    return model


def get_model() -> YOLO:
    """
    Get cached singleton YOLO model.
    Falls back to Streamlit resource cache if running in Streamlit,
    otherwise uses module singleton.
    """
    global _MODEL_CACHE

    try:
        import streamlit as st

        # If inside Streamlit execution context, use st.cache_resource
        @st.cache_resource(show_spinner="Loading YOLOv8 Candy Model...")
        def _get_st_model(path_str: str) -> YOLO:
            return load_raw_model(Path(path_str))

        path = resolve_model_path()
        return _get_st_model(str(path))
    except (ImportError, Exception):
        # Fallback to in-process memory singleton (for unittest/CLI)
        if _MODEL_CACHE is None:
            _MODEL_CACHE = load_raw_model()
        return _MODEL_CACHE


def extract_class_mapping(model: Optional[YOLO] = None) -> Dict[int, str]:
    """
    Extract the actual class mapping from model.names as source of truth.
    Preserves all 20 class IDs and labels.
    """
    target_model = model or get_model()
    names = target_model.names

    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}
    return {i: name for i, name in enumerate(CANDY_CLASSES)}


def validate_class_mapping(mapping: Dict[int, str]) -> Tuple[bool, str]:
    """Verify class mapping integrity against expected 20 classes."""
    if len(mapping) != len(CANDY_CLASSES):
        return (
            False,
            f"Class count mismatch: model has {len(mapping)} classes, expected {len(CANDY_CLASSES)}.",
        )
    return True, ""

