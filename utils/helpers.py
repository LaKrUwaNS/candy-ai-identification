"""Formatting and general helper utilities."""

from datetime import datetime
import uuid


def generate_unique_id(prefix: str = "") -> str:
    """Generate short unique hexadecimal identifier."""
    raw_id = str(uuid.uuid4())[:8]
    return f"{prefix}{raw_id}" if prefix else raw_id


def format_timestamp(dt: datetime = None) -> str:
    """Format datetime into standard readable string."""
    target = dt or datetime.now()
    return target.strftime("%Y-%m-%d %H:%M:%S")


def format_percentage(ratio: float) -> str:
    """Format decimal ratio (0.0 to 1.0) into percentage string."""
    return f"{ratio * 100:.1f}%"


def format_ms(ms: float) -> str:
    """Format millisecond duration to 1 decimal place."""
    return f"{ms:.1f} ms"

