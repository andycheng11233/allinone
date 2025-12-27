#!/usr/bin/env python3
"""
Shared utilities for the match analysis system.
Includes logging, color printing, text normalization, and constants.
"""

import logging
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Color output (optional)
try:
    from colorama import init as _init_colorama, Fore, Style
    _init_colorama(autoreset=True)
    COLORS_AVAILABLE = True
except Exception:
    COLORS_AVAILABLE = False

    class _Dummy:
        GREEN = RED = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ''
        BRIGHT = NORMAL = ''
    Fore = _Dummy()
    Style = _Dummy()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("playwright").setLevel(logging.WARNING)
logger = logging.getLogger("matcher")

# Constants
BAD_STRING = "暫無數據"
MIN_SECTIONS_FOR_FULL = 3  # Titan stats: minimum sections to mark as "full"

EXPECTED_TOP_LEVEL_SECTIONS = [
    "match", "league_standings", "data_comparison_recent10", "lineup_and_injuries",
    "last_match_player_ratings", "recent10_ratings_parsed", "future_matches",
    "head_to_head_sample", "league_trend_and_other_stats"
]

# Cache paths
AI_CACHE_PATH = Path(".cache/ai_processed.json")
HKJC_ODDS_PROCESSED_PATH = Path(".cache/hkjc_odds_processed.json")
TITAN_STATS_PROCESSED_PATH = Path(".cache/titan_stats_processed.json")

# Data paths
TITAN_STATS_BASE = Path("titan/stats")
GAME_FILE = "game.json"  # for Titan league filter


def cprint(text: str, color: str = '', style: str = ''):
    """Print colored text if colorama is available."""
    if COLORS_AVAILABLE:
        print(f"{style}{color}{text}{Style.RESET_ALL}")
    else:
        print(text)


def strip_accents(text: str) -> str:
    """Remove accents from text."""
    return ''.join(ch for ch in unicodedata.normalize('NFKD', text) if not unicodedata.combining(ch))


def _norm(val: Any) -> str:
    """Normalize a value to a string, handling None and null-like values."""
    s = str(val).strip() if val is not None else ""
    return "" if s.lower() in {"", "null", "none", "undefined"} else s


def find_best_float_in_text(text: str, min_val: float = -1e9, max_val: float = 1e9) -> Optional[float]:
    """Find the first valid float in text within the specified range."""
    if not text:
        return None
    tokens = re.findall(r'\d+\.\d+|\d+', text)
    for t in tokens:
        try:
            v = float(t)
        except ValueError:
            continue
        if min_val <= v <= max_val:
            return v
    return None


def parse_decimal_tokens_from_concatenated(text: str) -> List[float]:
    """Parse decimal values from concatenated text."""
    if not text:
        return []
    tokens = re.findall(r'\d{1,2}\.\d{1,2}', text)
    floats = []
    for t in tokens:
        try:
            v = float(t)
            if 0.0 <= v <= 10.0:
                floats.append(v)
        except ValueError:
            continue
    return floats


def extract_ratings_or_average_from_text(page_text: str) -> Tuple[Optional[float], List[float]]:
    """Extract ratings or averages from page text."""
    if not page_text:
        return None, []
    m = re.search(r'平均評分[:：]?\s*([0-9]{1,2}\.[0-9]{1,2})', page_text)
    if m:
        try:
            val = float(m.group(1))
            if 0.0 <= val <= 10.0:
                return val, [val]
        except Exception:
            pass
    m2 = re.search(r'(?:主隊|客隊)?近10場平均評分[:：]?\s*([0-9\.\s]{5,200})', page_text)
    if m2:
        snippet = m2.group(1)
        parsed = parse_decimal_tokens_from_concatenated(snippet)
        if parsed:
            avg = sum(parsed) / len(parsed)
            return avg, parsed
    all_decimals = parse_decimal_tokens_from_concatenated(page_text)
    if all_decimals:
        chosen = all_decimals[:10]
        avg = sum(chosen) / len(chosen)
        return avg, chosen
    return None, []
