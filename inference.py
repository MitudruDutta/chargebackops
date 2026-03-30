"""Challenge-compatible inference entry point (root re-export).

The submission contract requires inference.py at the repository root.
All logic lives in runners/inference.py.
"""

from runners.inference import main, run_inference  # noqa: F401

if __name__ == "__main__":
    main()
