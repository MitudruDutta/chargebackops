"""Live-provider audit for ChargebackOps."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline_runner import run_baseline
from inference import run_inference


def main() -> None:
    report = {
        "config": {
            "baseline_provider": os.getenv("BASELINE_PROVIDER"),
            "baseline_model": os.getenv("BASELINE_MODEL"),
            "api_base_url": os.getenv("API_BASE_URL"),
            "model_name": os.getenv("MODEL_NAME"),
            "strict_llm_mode": os.getenv("STRICT_LLM_MODE", ""),
        },
        "baseline": run_baseline().model_dump(),
        "inference": run_inference().model_dump(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
