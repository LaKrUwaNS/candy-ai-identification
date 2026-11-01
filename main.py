"""Candy AI – Intelligent Candy Identification System
Launcher script to start the Streamlit application.
"""

import sys
import subprocess
from pathlib import Path


def main() -> None:
    app_path = Path(__file__).resolve().parent / "app.py"
    print("=" * 60)
    print("  🍬 Starting Candy AI – Intelligent Candy Identification System")
    print("=" * 60)
    print(f"Target application script: {app_path}")
    print("Starting Streamlit server on http://localhost:8501 ...\n")

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nCandy AI application stopped by user.")


if __name__ == "__main__":
    main()
