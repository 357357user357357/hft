"""Configuration loader — JSON config files override constants.py defaults.

Usage:
    python main.py --config my_experiment.json --file data/BTCUSDT.zip

Example config (config_example.json):
    {
        "SHOT_DISTANCE_PCT": 0.12,
        "SHOT_TP_PCT": 0.08,
        "DEFAULT_ORDER_SIZE_USDT": 200.0
    }

Only keys that match constants.py names are applied; unknown keys are warned.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import constants

logger = logging.getLogger("hft")


def load_config(path: Optional[str]) -> None:
    """Load a JSON config file and override matching constants.py values."""
    if path is None:
        return

    config_path = Path(path)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return

    with open(config_path, "r") as f:
        overrides = json.load(f)

    if not isinstance(overrides, dict):
        logger.error("Config file must contain a JSON object, got %s", type(overrides).__name__)
        return

    applied = 0
    for key, value in overrides.items():
        if hasattr(constants, key):
            old = getattr(constants, key)
            setattr(constants, key, type(old)(value))
            logger.debug("Config override: %s = %s (was %s)", key, value, old)
            applied += 1
        else:
            logger.warning("Unknown config key: %s (ignored)", key)

    logger.info("Applied %d config overrides from %s", applied, config_path)


def save_current_config(path: str) -> None:
    """Save all current constants to a JSON file (for reproducibility)."""
    data = {}
    for key in sorted(dir(constants)):
        if key.isupper() and not key.startswith("_"):
            data[key] = getattr(constants, key)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved current config to %s", path)
