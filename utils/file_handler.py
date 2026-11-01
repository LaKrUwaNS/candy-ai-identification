"""File and path management utilities for Candy AI."""

from pathlib import Path
from typing import Tuple

from config.settings import MODEL_PATH, PROJECT_ROOT, RELATIVE_MODEL_PATH


def resolve_model_path() -> Path:
    """
    Resolve model path robustly across different working directories.
    Checks explicit settings path, current working directory, and relative root.
    """
    # 1. Configured absolute path
    if MODEL_PATH.is_file():
        return MODEL_PATH.resolve()

    # 2. CWD-relative path
    cwd_path = Path(RELATIVE_MODEL_PATH).resolve()
    if cwd_path.is_file():
        return cwd_path

    # 3. Project root relative path
    root_candidate = (PROJECT_ROOT / RELATIVE_MODEL_PATH).resolve()
    if root_candidate.is_file():
        return root_candidate

    # 4. Search upward
    curr = Path(__file__).resolve().parent
    for _ in range(4):
        check = curr / "Model" / "best.pt"
        if check.is_file():
            return check.resolve()
        curr = curr.parent

    # Return default configured path even if missing
    return MODEL_PATH.resolve()


def check_model_exists() -> Tuple[bool, str]:
    """Validate existence of trained model weights file."""
    path = resolve_model_path()
    if path.is_file():
        return True, str(path)
    return False, (
        f"Model file not found at '{RELATIVE_MODEL_PATH}' (resolved to '{path}'). "
        "Please ensure 'Model/best.pt' is present in your project directory."
    )


def get_file_size_mb(path: Path) -> float:
    """Return size of file in megabytes."""
    if not path.is_file():
        return 0.0
    return path.stat().st_size / (1024 * 1024)

