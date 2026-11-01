"""Centralized application configuration for Candy AI.
Contains paths, constants, thresholds, class definitions, and color themes.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple

# -----------------------------------------------------------------------------
# Base Application Information
# -----------------------------------------------------------------------------
APP_TITLE: str = "Candy AI – Intelligent Candy Identification System"
APP_SUBTITLE: str = "Automated Confectionery Detection, Localization & Counting"
APP_ICON: str = "🍬"
VERSION: str = "2.0.0"

# -----------------------------------------------------------------------------
# Path Configurations
# -----------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RELATIVE_MODEL_PATH: str = "Model/best.pt"
MODEL_PATH: Path = PROJECT_ROOT / "Model" / "best.pt"
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
LOGO_PATH: Path = ASSETS_DIR / "logo.png"
STYLES_PATH: Path = ASSETS_DIR / "styles.css"

# -----------------------------------------------------------------------------
# Inference Thresholds & Parameter Defaults
# -----------------------------------------------------------------------------
DEFAULT_CONFIDENCE: float = 0.25
DEFAULT_IOU: float = 0.45
DEFAULT_IMAGE_SIZE: int = 640

MIN_CONFIDENCE: float = 0.05
MAX_CONFIDENCE: float = 1.00
CONFIDENCE_STEP: float = 0.05

MIN_IOU: float = 0.10
MAX_IOU: float = 0.95
IOU_STEP: float = 0.05

SUPPORTED_RESOLUTIONS: List[int] = [320, 480, 640, 800, 1024, 1280]

# -----------------------------------------------------------------------------
# Image Handling & Input Restrictions
# -----------------------------------------------------------------------------
ALLOWED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
MIN_IMAGE_DIMENSION: int = 10
MAX_IMAGE_DIMENSION: int = 8192
DEFAULT_THUMBNAIL_SIZE: Tuple[int, int] = (160, 160)

# -----------------------------------------------------------------------------
# Supported Candy Classes (20 Custom Classes)
# Source of truth must be verified with model.names
# -----------------------------------------------------------------------------
CANDY_CLASSES: List[str] = [
    "Bounty",
    "Dare_caramel",
    "Dare_cocoa",
    "MMs_peanut",
    "MMs_regular",
    "Mars",
    "Twix_caramel",
    "Twix_clasic",
    "Twix_white",
    "airheads",
    "eugenia",
    "gummy_worms",
    "milky_way",
    "nerds",
    "skittles",
    "snickers",
    "snickers_creamy",
    "starbust",
    "three_musketeers",
    "twizzlers",
]

# -----------------------------------------------------------------------------
# 20 Distinct High-Contrast Accessible Colors for Classes (RGB)
# -----------------------------------------------------------------------------
CLASS_COLOR_PALETTE: List[Tuple[int, int, int]] = [
    (0, 168, 150),    # 0: Bounty (Deep Turquoise)
    (217, 119, 6),    # 1: Dare_caramel (Warm Amber)
    (139, 69, 19),    # 2: Dare_cocoa (Chocolate Brown)
    (245, 158, 11),   # 3: MMs_peanut (Peanut Gold)
    (239, 68, 68),    # 4: MMs_regular (Bright Red)
    (190, 18, 60),    # 5: Mars (Crimson Ruby)
    (234, 179, 8),    # 6: Twix_caramel (Golden Caramel)
    (180, 83, 9),     # 7: Twix_clasic (Bronze Gold)
    (100, 116, 139),  # 8: Twix_white (Slate Silver)
    (59, 130, 246),   # 9: airheads (Electric Blue)
    (249, 115, 22),   # 10: eugenia (Orange Peel)
    (16, 185, 129),   # 11: gummy_worms (Lime Emerald)
    (99, 102, 241),   # 12: milky_way (Royal Indigo)
    (139, 92, 246),   # 13: nerds (Vivid Violet)
    (244, 63, 94),    # 14: skittles (Bright Coral)
    (120, 53, 15),    # 15: snickers (Saddle Brown)
    (202, 138, 4),    # 16: snickers_creamy (Cream Ochre)
    (236, 72, 153),   # 17: starbust (Hot Pink)
    (2, 132, 199),    # 18: three_musketeers (Midnight Blue)
    (153, 27, 27),    # 19: twizzlers (Cherry Red)
]


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB integer tuple to Hex color string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


CLASS_COLOR_HEX: List[str] = [rgb_to_hex(c) for c in CLASS_COLOR_PALETTE]


def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
    """Retrieve consistent RGB color tuple for a class ID."""
    return CLASS_COLOR_PALETTE[class_id % len(CLASS_COLOR_PALETTE)]


def get_hex_for_class(class_id: int) -> str:
    """Retrieve consistent Hex color string for a class ID."""
    return CLASS_COLOR_HEX[class_id % len(CLASS_COLOR_HEX)]

