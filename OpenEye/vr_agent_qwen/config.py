"""
vr_agent/config.py
------------------
Global configuration: model names, paths, and UI state constants.
"""

from pathlib import Path
import os

# ── Model Names ───────────────────────────────────────────────────────────────
MODEL_PLANNER = os.getenv("MODEL_PLANNER", "gemini-3-flash-preview")
# Qwen grounding is enabled by default for this package.
MODEL_GROUNDING = os.getenv("MODEL_GROUNDING", "qwen3")
MODEL_VERIFICATION = os.getenv("MODEL_VERIFICATION", "gemini-2.5-flash")
MODEL_DESCRIPTION = os.getenv("MODEL_DESCRIPTION", "gemini-2.5-flash-lite-preview-09-2025")
MODEL_WHITE_CANE = os.getenv("MODEL_WHITE_CANE", "gemini-3-flash-preview")

# ── Whisper STT ───────────────────────────────────────────────────────────────
WHISPER_MODEL = "small.en"

# ── Paths ─────────────────────────────────────────────────────────────────────
LOG_DIR = Path("agent_logs_v2")

# ── Feature Flags ─────────────────────────────────────────────────────────────
SHOW_VISION_PREVIEW = False


# ── Voice Menu States ─────────────────────────────────────────────────────────
class VoiceMenuState:
    IDLE          = "idle"
    MAIN_MENU     = "main_menu"
    WHITE_CANE_MENU = "white_cane_menu"
    CONFIRMATION  = "confirmation"
