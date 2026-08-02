#!/usr/bin/env python3
"""Create a .venv virtual environment and install requirements.txt into it.

Works on Linux, macOS, and Windows (where the venv layout is Scripts\\
instead of bin/). Run with whatever Python you want the venv to use:

    python3 setup_venv.py        # Linux / macOS
    py setup_venv.py             # Windows
"""

import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"

# PyPI's Windows torch wheels are CPU-only; the CUDA builds live on
# pytorch.org's own index. cu128 covers Ada and Blackwell GPUs alike.
# (Linux/macOS need nothing special: PyPI wheels already include CUDA / MPS.)
WINDOWS_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def main() -> None:
    if VENV.is_dir():
        print(f"Virtual environment already exists at {VENV}")
    else:
        print(f"Creating virtual environment at {VENV}")
        venv.create(VENV, with_pip=True)

    python = str(venv_python())
    print("Installing dependencies from requirements.txt")
    subprocess.run([python, "-m", "pip", "install", "--upgrade", "pip"],
                   check=True)

    # On Windows with an NVIDIA GPU, install CUDA torch first so the
    # requirements step's "torch>=2.0" is already satisfied by it.
    if sys.platform == "win32" and shutil.which("nvidia-smi"):
        print("NVIDIA GPU detected -- installing CUDA-enabled torch")
        subprocess.run([python, "-m", "pip", "install", "torch",
                        "--index-url", WINDOWS_CUDA_INDEX], check=True)
    subprocess.run([python, "-m", "pip", "install", "-r",
                    str(ROOT / "requirements.txt")], check=True)

    print("\nDone. Activate the environment with:")
    if sys.platform == "win32":
        print(r"  .venv\Scripts\Activate.ps1    (PowerShell)")
        print(r"  .venv\Scripts\activate.bat    (cmd)")
    else:
        print("  source .venv/bin/activate")


if __name__ == "__main__":
    main()
