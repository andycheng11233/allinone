#!/usr/bin/env python3
"""
Unified MacauSlot + Titan007 + HKJC matcher + AI analysis.

Key points:
- HKJC bulk All-Odds scrape runs first (clicks “顯示更多” on home page). Results saved in hkjc/odds/hkjc_allodds_<ts>.json.
- HKJC home scraper extracts event_ids via href, data-attrs, and tvChannels classes.
- Reuses cached HKJC odds and Titan stats across runs (skip re-scraping already processed IDs).
- Titan stats scraper and AI pipeline are restored; per-match AI status/reason recorded.
- Comparison export is 1:1 (no Cartesian explosion) and saved to Excel.
- AI results are saved to Excel in addition to JSON.
- New: All-sources Excel export with matched rows first, then unmatched (HKJC/Titan/Macau).

Setup:
  pip install -r requirements.txt
  playwright install
  export DEEPSEEK_API_KEY=...
  export DEEPSEEK_API_URL=https://api.deepseek.com/chat/completions
Run:
  python3 allinone.py
"""

import asyncio
import os
import re
import json
import logging
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import httpx
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup, Tag
import pandas as pd

# Debug instrumentation (optional)
try:
    from debug_instrumentation import (
        init_debug_session,
        save_rendered_html,
        save_parsed_json,
        log_mapping_decision,
        log_info
    )
    DEBUG_INSTRUMENTATION_AVAILABLE = True
except Exception:
    DEBUG_INSTRUMENTATION_AVAILABLE = False

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("matcher")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
DEEPSEEK_TIMEOUT = float(os.getenv("DEEPSEEK_TIMEOUT", "30"))
DEEPSEEK_RETRIES = int(os.getenv("DEEPSEEK_RETRIES", "3"))

if not DEEPSEEK_API_KEY:
    logger.warning("DEEPSEEK_API_KEY not set. AI functionality will fail unless you set the env var.")

AI_CACHE_PATH = Path(".cache/ai_processed.json")
HKJC_ODDS_PROCESSED_PATH = Path(".cache/hkjc_odds_processed.json")
TITAN_STATS_PROCESSED_PATH = Path(".cache/titan_stats_processed.json")


def load_ai_cache() -> Dict[str, Any]:
    try:
        if AI_CACHE_PATH.exists():
            return json.loads(AI_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load AI cache: %s", e)
    return {}


def save_ai_cache(cache: Dict[str, Any]):
    try:
        AI_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        AI_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save AI cache: %s", e)


def load_cache_set(path: Path) -> set:
    try:
        if path.exists():
            return set(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        pass
    return set()


def save_cache_set(path: Path, data: set):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(list(data))), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save cache set to %s: %s", path, e)


def cprint(text: str, color: str = '', style: str = ''):
    if COLORS_AVAILABLE:
        print(f"{style}{color}{text}{Style.RESET_ALL}")
    else:
        print(text)

# ---------------------------------------------------------------------------
# Robust numeric helpers
# ---------------------------------------------------------------------------
def find_best_float_in_text(text: str, min_val: float = -1e9, max_val: float = 1e9) -> Optional[float]:
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

# ---------------------------------------------------------------------------
# DeepSeek async client
# ---------------------------------------------------------------------------
async def call_deepseek_api_async(prompt: str, timeout: int = None, max_retries: int = None) -> str:
    if timeout is None:
        timeout = int(DEEPSEEK_TIMEOUT)
    if max_retries is None:
        max_retries = int(DEEPSEEK_RETRIES)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}" if DEEPSEEK_API_KEY else ""
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000
    }

    backoff_base = 0.6
    last_err = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.post(DEEPSEEK_API_URL, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict):
                    choices = data.get("choices")
                    if choices and isinstance(choices, list) and len(choices) > 0:
                        first = choices[0]
                        if isinstance(first, dict):
                            msg = first.get("message") or first.get("text") or {}
                            if isinstance(msg, dict):
                                content = msg.get("content") or msg.get("text") or ""
                            elif isinstance(msg, str):
                                content = msg
                            else:
                                content = ""
                        else:
                            content = str(first)
                        return content.strip()
                return resp.text
            except httpx.HTTPStatusError as e:
                last_err = str(e)
                status = getattr(e.response, "status_code", None)
                logger.error("DeepSeek HTTP error (attempt %d): %s", attempt, e)
                if status and status < 500 and status != 429:
                    break
            except Exception as e:
                last_err = str(e)
                logger.error("DeepSeek request failed (attempt %d): %s", attempt, e)
            await asyncio.sleep(backoff_base * attempt)
    raise RuntimeError(f"DeepSeek API calls failed: {last_err}")

# ---------------------------------------------------------------------------
# AI integration helpers (with merge of sections)
# ---------------------------------------------------------------------------
EXPECTED_TOP_LEVEL_SECTIONS = [
    "match", "league_standings", "data_comparison_recent10", "lineup_and_injuries",
    "last_match_player_ratings", "recent10_ratings_parsed", "future_matches",
    "head_to_head_sample", "league_trend_and_other_stats"
]

def normalize_parsed_data(parsed: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(parsed)
    if isinstance(parsed.get("sections"), dict):
        for k, v in parsed["sections"].items():
            if k not in merged or merged.get(k) is None:
                merged[k] = v

    normalized: Dict[str, Any] = {}
    missing: List[str] = []
    available: List[str] = []

    for key in EXPECTED_TOP_LEVEL_SECTIONS:
        val = merged.get(key)
        if val is None:
            normalized[key] = None
            missing.append(key)
        else:
            normalized[key] = val
            available.append(key)

    match_block = merged.get("match") or {}
    normalized["match"] = {
        "home_team": match_block.get("home_team") or merged.get("home_team"),
        "away_team": match_block.get("away_team") or merged.get("away_team"),
        "competition": match_block.get("competition") or merged.get("competition"),
        "datetime": match_block.get("datetime") or merged.get("datetime"),
        "venue": match_block.get("venue") or merged.get("venue"),
    }

    rr = merged.get("recent10_ratings_parsed") or {}
    normalized["recent10_ratings_parsed"] = {
        "home_recent_ratings_raw": rr.get("home_recent_ratings_raw"),
        "home_recent_ratings": list(rr.get("home_recent_ratings") or []),
        "home_recent_average": rr.get("home_recent_average") or merged.get("home_rating"),
        "away_recent_ratings_raw": rr.get("away_recent_ratings_raw"),
        "away_recent_ratings": list(rr.get("away_recent_ratings") or []),
        "away_recent_average": rr.get("away_recent_average") or merged.get("away_rating"),
    }

    section_counts = {}
    for key in EXPECTED_TOP_LEVEL_SECTIONS:
        v = normalized.get(key)
        if isinstance(v, list):
            section_counts[key] = len(v)
        elif isinstance(v, dict):
            section_counts[key] = len(v.keys())
        elif v is None:
            section_counts[key] = 0
        else:
            section_counts[key] = 1

    normalized["_meta"] = {
        "missing_fields": missing,
        "available_sections": available,
        "section_counts": section_counts
    }
    return normalized

def has_meaningful_data_for_ai(normalized_stats: Dict[str, Any]) -> bool:
    meta = normalized_stats.get("_meta", {})
    available = meta.get("available_sections", [])
    if available:
        return True
    rr = normalized_stats.get("recent10_ratings_parsed") or {}
    if rr.get("home_recent_ratings") or rr.get("away_recent_ratings") or rr.get("home_recent_average") or rr.get("away_recent_average"):
        return True
    return False

def build_ai_prompt_with_availability(normalized_data: Dict[str, Any], use_chinese: bool = True) -> str:
    meta = normalized_data.get("_meta", {})
    available = meta.get("available_sections", [])
    missing = meta.get("missing_fields", [])

    def render_section(name: str, max_chars: int = 800) -> str:
        sec = normalized_data.get(name)
        if not sec:
            return ""
        try:
            s = json.dumps(sec, ensure_ascii=False, indent=0)
        except Exception:
            s = str(sec)
        return s[:max_chars] + ("..." if len(s) > max_chars else "")

    available_str = ", ".join(available) if available else "none"
    missing_str = ", ".join(missing) if missing else "none"

    excerpts = ""
    for sec_name in ("recent10_ratings_parsed", "league_standings", "data_comparison_recent10",
                     "last_match_player_ratings", "lineup_and_injuries"):
        excerpt = render_section(sec_name)
        if excerpt:
            excerpts += f"\n\n=== {sec_name} ===\n{excerpt}"

    if use_chinese:
        header = (
            f"你是一位資深足球博彩分析師。以下資料已由爬蟲解析和標準化，系統說明哪些欄位存在或遺漏。\n"
            f"Available sections: {available_str}\n"
            f"Missing sections: {missing_str}\n\n"
            "請基於可用數據（若某些欄位遺失，請明確指出）推薦一個最有價值的投注選項，並以JSON格式回覆，"
            "僅包含如下字段：\n"
            '{ "best_bet_market": "投注市場", "best_bet_selection": "具體選擇", "confidence_level": "1-10", "brief_reasoning": "簡短原因" }\n\n'
            "只輸出JSON，不要其他文字。"
        )
    else:
        header = (
            f"You are an experienced football betting analyst. The parser produced the following available/missing sections.\n"
            f"Available: {available_str}\nMissing: {missing_str}\n\n"
            "Based on available data (explicitly note missing fields if they matter), recommend a single best bet in JSON:\n"
            '{ "best_bet_market": "...", "best_bet_selection": "...", "confidence_level": "1-10", "brief_reasoning": "..." }\n'
            "Output only JSON."
        )

    prompt = header + excerpts
    return prompt

def parse_ai_json_response(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not text:
        return None, None
    try:
        s = text.strip()
        if s.startswith("{") and s.endswith("}"):
            return json.loads(s), s
        # Find JSON object - use a simpler pattern to avoid ReDoS
        start = text.find('{')
        if start != -1:
            # Find matching closing brace by counting braces
            depth = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i+1]
                        return json.loads(candidate), candidate
    except Exception as e:
        logger.debug("AI JSON parse failed: %s", e)
    try:
        fallback = {}
        market = re.search(r'"?best_bet_market"?\s*[:=]\s*"([^"]+)"', text)
        selection = re.search(r'"?best_bet_selection"?\s*[:=]\s*"([^"]+)"', text)
        confidence = re.search(r'"?confidence_level"?\s*[:=]\s*([0-9]+)', text)
        reasoning = re.search(r'"?brief_reasoning"?\s*[:=]\s*"([^"]+)"', text)
        if market:
            fallback["best_bet_market"] = market.group(1)
        if selection:
            fallback["best_bet_selection"] = selection.group(1)
        if confidence:
            fallback["confidence_level"] = int(confidence.group(1))
        if reasoning:
            fallback["brief_reasoning"] = reasoning.group(1)
        if fallback:
            return fallback, json.dumps(fallback, ensure_ascii=False)
    except Exception:
        pass
    return None, None

async def perform_ai_analysis_for_match_async(
    normalized_stats: Dict[str, Any],
    call_deepseek_api_async_fn,
    max_retries: int = 2,
    short_circuit_when_no_data: bool = True
) -> Dict[str, Any]:
    result = {
        "best_bet_market": "No Data",
        "best_bet_selection": "No Analysis Available",
        "confidence_level": 0,
        "brief_reasoning": "Insufficient statistical data available for analysis.",
        "ai_raw_response": None,
        "ai_parsed_json": None,
        "data_availability": normalized_stats.get("_meta", {})
    }

    if short_circuit_when_no_data and not has_meaningful_data_for_ai(normalized_stats):
        logger.info("Short-circuiting AI call: no meaningful sections or ratings present")
        return result

    prompt = build_ai_prompt_with_availability(normalized_stats, use_chinese=True)
    if DEBUG_INSTRUMENTATION_AVAILABLE:
        try:
            save_parsed_json("ai_prompt", normalized_stats.get("match", {}).get("home_team", "unknown"), {"prompt": prompt[:4000]})
            log_info("AI prompt built", {"available_sections": normalized_stats.get("_meta", {})})
        except Exception:
            pass

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            ai_text = await call_deepseek_api_async_fn(prompt)
            result["ai_raw_response"] = ai_text
            if DEBUG_INSTRUMENTATION_AVAILABLE:
                try:
                    save_parsed_json("ai_response_raw", normalized_stats.get("match", {}).get("home_team", "unknown"), {"raw": ai_text[:4000]})
                except Exception:
                    pass
            parsed, raw = parse_ai_json_response(ai_text)
            if parsed:
                required = ["best_bet_market", "best_bet_selection", "confidence_level", "brief_reasoning"]
                if all(k in parsed for k in required):
                    result.update({
                        "best_bet_market": parsed.get("best_bet_market"),
                        "best_bet_selection": parsed.get("best_bet_selection"),
                        "confidence_level": parsed.get("confidence_level"),
                        "brief_reasoning": parsed.get("brief_reasoning"),
                        "ai_parsed_json": parsed
                    })
                    return result
                else:
                    result["ai_parsed_json"] = parsed
                    result["brief_reasoning"] = "AI returned partial result; missing keys"
                    result["confidence_level"] = parsed.get("confidence_level", 1)
                    return result
            logger.warning("AI response contained no parsable JSON (attempt %d).", attempt)
            last_err = "No JSON in response"
            await asyncio.sleep(0.6 * attempt)
        except Exception as e:
            last_err = str(e)
            logger.error("AI call attempt %d failed: %s", attempt, e)
            await asyncio.sleep(0.6 * attempt)
    result["brief_reasoning"] = f"AI call failed: {last_err}"
    result["confidence_level"] = 0
    return result

# ---------------------------------------------------------------------------
# HKJC All-Odds parsers and scraper
# ---------------------------------------------------------------------------
AO_DISPLAY_ZH = {
    "HAD": "主客和", "FHA": "半場主客和", "HHA": "讓球主客和", "HHA_Extra": "讓球主客和",
    "HDC": "讓球", "HIL": "入球大細", "FHL": "半場入球大細",
    "CHL": "開出角球大細", "FCH": "半場開出角球大細",
    "CHD": "開出角球讓球", "FHC": "半場開出角球讓球",
    "CRS": "波膽", "FCS": "半場波膽", "FTS": "第一隊入球",
    "TTG": "總入球", "OOE": "入球單雙", "HFT": "半全場",
    "FGS": "首名入球", "LGS": "最後入球球員", "AGS": "任何時間入球球員",
    "MSP": "特別項目",
}

def ao_clean_text(el):
    return el.get_text(strip=True) if el else None

def ao_clean_odds_text(span):
    if not span:
        return None
    text = span.get_text(strip=True)
    cleaned = re.sub(r"[^\d.]", "", text)
    return float(cleaned) if cleaned else None

def ao_next_match_row_container(coupon):
    if not coupon:
        return None
    sib = coupon.find_next_sibling()
    while sib is not None and not ("match-row-container" in sib.get("class", [])):
        sib = sib.find_next_sibling()
    return sib

def ao_parse_allodds_match_header(soup):
    mi = soup.select_one(".match-info")
    if not mi:
        return {}
    match_id = ao_clean_text(mi.select_one(".match .val"))
    home = ao_clean_text(mi.select_one(".team .home"))
    away = ao_clean_text(mi.select_one(".team .away"))
    time_raw = ao_clean_text(mi.select_one(".time .val"))
    timg = mi.select_one(".matchInfoTourn img")
    tournament = timg["title"] if timg and timg.has_attr("title") else None
    return {
        "match_id": match_id,
        "home_team": home,
        "away_team": away,
        "tournament": tournament,
        "time_raw": time_raw,
    }

def ao_parse_had_like_from_row(row, odds_class):
    odds_block = row.select_one(f".odds.{odds_class}")
    if not odds_block:
        return None
    grids = odds_block.select(".oddsCheckboxGrid")
    if len(grids) < 3:
        return None
    co = ao_clean_odds_text
    return {
        "home_odds": co(grids[0].select_one(".add-to-slip")),
        "draw_odds": co(grids[1].select_one(".add-to-slip")),
        "away_odds": co(grids[2].select_one(".add-to-slip")),
    }

def ao_parse_hha_multi_lines_from_row(row):
    res = []
    odds_line = row.select_one(".oddsLine.HHA")
    if not odds_line:
        return res
    line_blocks = odds_line.select(".odds.show")
    for line_index, line_block in enumerate(line_blocks):
        items = line_block.select(".hdcOddsItem")
        if len(items) != 3:
            continue

        def cond(item):
            c = item.select_one(".cond")
            return ao_clean_text(c).strip("[]") if c else ""

        home_hcap = cond(items[0]); draw_hcap = cond(items[1]); away_hcap = cond(items[2])
        co = ao_clean_odds_text
        home_odds = co(items[0].select_one(".add-to-slip"))
        draw_odds = co(items[1].select_one(".add-to-slip"))
        away_odds = co(items[2].select_one(".add-to-slip"))
        give_home = None
        if home_hcap.startswith("-"):
            give_home = True
        elif away_hcap.startswith("-"):
            give_home = False
        else:
            if home_odds is not None and away_odds is not None:
                give_home = home_odds < away_odds
        market_type = "HHA" if line_index == 0 else "HHA_Extra"
        res.append({
            "market_type": market_type,
            "line_index": line_index + 1,
            "home_odds": home_odds, "draw_odds": draw_odds, "away_odds": away_odds,
            "euro_handicap_value": home_hcap,
            "euro_handicap_give_home": give_home,
        })
    return res

def ao_parse_hdc_from_row(row):
    odds_line = row.select_one(".oddsLine.HDC")
    if not odds_line:
        return None
    lb = odds_line.select_one(".odds.show")
    if not lb:
        return None
    items = lb.select(".hdcOddsItem")
    if len(items) != 2:
        return None

    def cond(item):
        c = item.select_one(".cond")
        return ao_clean_text(c).strip("[]") if c else ""

    home_hcap = cond(items[0]); away_hcap = cond(items[1])
    co = ao_clean_odds_text
    home_odds = co(items[0].select_one(".add-to-slip"))
    away_odds = co(items[1].select_one(".add-to-slip"))
    give_home = None
    if home_hcap.startswith("-"):
        give_home = True
    elif away_hcap.startswith("-"):
        give_home = False
    else:
        if home_odds is not None and away_odds is not None:
            give_home = home_odds < away_odds
    return {
        "asia_handicap_value": home_hcap,
        "asia_handicap_give_home": give_home,
        "home_odds": home_odds, "away_odds": away_odds,
    }

def ao_parse_ou_market_from_row(row, class_name, goal_field_name="goal_line"):
    odds_line = row.select_one(f".oddsLine.{class_name}")
    if not odds_line:
        return []
    res = []
    line_nums = odds_line.select(".lineNum.show")
    odds_blocks = odds_line.select(".odds.show")
    for line_idx, (ln, ob) in enumerate(zip(line_nums, odds_blocks)):
        line_text = ao_clean_text(ln).strip("[]") if ln else None
        grids = ob.select(".oddsCheckboxGrid")
        if len(grids) < 2:
            continue
        co = ao_clean_odds_text
        over_odds = co(grids[0].select_one(".add-to-slip"))
        under_odds = co(grids[1].select_one(".add-to-slip"))
        res.append({
            "line_index": line_idx + 1,
            goal_field_name: line_text,
            "over_odds": over_odds,
            "under_odds": under_odds,
        })
    return res

def ao_parse_crs_matrix(row):
    res = []
    for odds_cell in row.select(".crsTable .odds"):
        score = ao_clean_text(odds_cell.select_one(".crsSel"))
        odds = ao_clean_odds_text(odds_cell.select_one(".add-to-slip"))
        if score and odds is not None:
            res.append({"score": score, "odds": odds})
    return res

def ao_parse_fts(row):
    odds_block = row.select_one(".oddsFTS")
    if not odds_block:
        return None
    grids = odds_block.select(".oddsCheckboxGrid")
    if len(grids) < 3:
        return None
    co = ao_clean_odds_text
    return {
        "home_first": co(grids[0].select_one(".add-to-slip")),
        "no_goal": co(grids[1].select_one(".add-to-slip")),
        "away_first": co(grids[2].select_one(".add-to-slip")),
    }

def ao_parse_ttg(row):
    odds_block = row.select_one(".oddsTTG")
    if not odds_block:
        return []
    res = []
    for block in odds_block.find_all("div", recursive=False):
        goals = ao_clean_text(block.select_one(".goals-number"))
        odds = ao_clean_odds_text(block.select_one(".add-to-slip"))
        if goals and odds is not None:
            res.append({"goals": goals, "odds": odds})
    return res

def ao_parse_ooe(row):
    odds_block = row.select_one(".oddsOOE")
    if not odds_block:
        return None
    grids = odds_block.select(".oddsCheckboxGrid")
    if len(grids) < 2:
        return None
    co = ao_clean_odds_text
    return {
        "odd": co(grids[0].select_one(".add-to-slip")),
        "even": co(grids[1].select_one(".add-to-slip")),
    }

def ao_parse_hft(row):
    odds_block = row.select_one(".oddsHFT")
    if not odds_block:
        return []
    res = []
    for block in odds_block.find_all("div", recursive=False):
        label = ao_clean_text(block.select_one(".goals-number"))
        odds = ao_clean_odds_text(block.select_one(".add-to-slip"))
        if label and odds is not None:
            res.append({"combo": label, "odds": odds})
    return res

def ao_parse_scorer_market(row):
    res = []
    grids = row.select(".oddsCheckboxGrid")

    def pull_candidates(grid):
        cands = []
        for sib in grid.previous_siblings:
            if getattr(sib, "get_text", None):
                cands.append(ao_clean_text(sib))
        parent = grid.find_parent(["td", "div"])
        if parent:
            for sib in parent.find_previous_siblings():
                if getattr(sib, "get_text", None):
                    cands.append(ao_clean_text(sib))
        for prev in grid.find_all_previous(["div", "td", "th", "span"], limit=8):
            txt = ao_clean_text(prev)
            if txt:
                cands.append(txt)
        return cands

    for g in grids:
        odds = ao_clean_odds_text(g.find_next("span", class_="add-to-slip"))
        gid = g.get("id", "") or ""
        m = re.search(r"_(\d{3,})", gid)
        code = m.group(1) if m else None
        name = None
        for txt in pull_candidates(g):
            if not txt:
                continue
            m2 = re.search(r"\b(\d{3})\b\s*([A-Za-z\u4e00-\u9fff].+)", txt)
            if m2:
                if not code:
                    code = m2.group(1)
                name = m2.group(2).strip()
                break
        res.append({"player_code": code, "player_name": name, "odds": odds})
    return res

def ao_parse_msp(row):
    raw = row.get_text(" ", strip=True)
    if not raw:
        return []
    items = []
    parts = re.split(r"項目編號[:：]\s*", raw)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m_id = re.match(r"(\d+)", part)
        item_id = m_id.group(1) if m_id else None
        m_q = re.split(r"\(\d+\)", part, maxsplit=1)
        if len(m_q) == 2:
            question = m_q[0].strip()
            rest = "(" + m_q[1]
        else:
            question = None
            rest = part
        options = []
        for opt_num, label, odds in re.findall(r"\((\d+)\)\s*([^(]+?)\s+(\d+(?:\.\d+)?)", rest):
            options.append({"option": opt_num, "label": label.strip(), "odds": float(odds)})
        items.append({"item_id": item_id, "question": question, "options": options, "raw": part})
    if not items:
        odds_list = [ao_clean_odds_text(span) for span in row.select(".add-to-slip")]
        items.append({"raw": raw, "odds": [o for o in odds_list if o is not None]})
    return items

def ao_parse_allodds_from_html(html: str):
    soup = BeautifulSoup(html, "html.parser")
    match_meta = ao_parse_allodds_match_header(soup)
    markets = {}

    def add_display(code, obj):
        if obj is None:
            return None
        if isinstance(obj, dict):
            obj["display_name_zh"] = AO_DISPLAY_ZH.get(code, "")
        return obj

    def add_display_list(code, lst):
        if lst is None:
            return None
        return [item | {"display_name_zh": AO_DISPLAY_ZH.get(code, "")} for item in lst]

    for code, cls, parser in [
        ("HAD", "couponHAD", lambda r: ao_parse_had_like_from_row(r, "oddsHAD")),
        ("FHA", "couponFHA", lambda r: ao_parse_had_like_from_row(r, "oddsFHA")),
        ("HDC", "couponHDC", ao_parse_hdc_from_row),
        ("CHD", "couponCHD", ao_parse_hdc_from_row),
        ("FHC", "couponFHC", ao_parse_hdc_from_row),
    ]:
        coupon = soup.select_one(f".coupon.{cls}")
        if coupon:
            row = ao_next_match_row_container(coupon)
            if row:
                markets[code] = add_display(code, parser(row))

    coupon = soup.select_one(".coupon.couponHHA")
    if coupon:
        row = ao_next_match_row_container(coupon)
        if row:
            hha_lines = ao_parse_hha_multi_lines_from_row(row)
            for line in hha_lines:
                line["display_name_zh"] = AO_DISPLAY_ZH.get(line["market_type"], "")
                markets.setdefault(line["market_type"], []).append(line)

    for code, cls in [("HIL", "couponHIL"), ("FHL", "couponFHL"), ("CHL", "couponCHL"), ("FCH", "couponFCH")]:
        coupon = soup.select_one(f".coupon.{cls}")
        if coupon:
            row = ao_next_match_row_container(coupon)
            if row:
                markets[code] = add_display_list(code, ao_parse_ou_market_from_row(row, code))

    for code, cls in [("CRS", "couponCRS"), ("FCS", "couponFCS")]:
        coupon = soup.select_one(f".coupon.{cls}")
        if coupon:
            row = ao_next_match_row_container(coupon)
            if row:
                markets[code] = {"display_name_zh": AO_DISPLAY_ZH[code], "scores": ao_parse_crs_matrix(row)}

    coupon = soup.select_one(".coupon.couponFTS")
    if coupon:
        row = ao_next_match_row_container(coupon)
        if row:
            markets["FTS"] = add_display("FTS", ao_parse_fts(row))
    coupon = soup.select_one(".coupon.couponTTG")
    if coupon:
        row = ao_next_match_row_container(coupon)
        if row:
            markets["TTG"] = {"display_name_zh": AO_DISPLAY_ZH["TTG"], "buckets": ao_parse_ttg(row)}
    coupon = soup.select_one(".coupon.couponOOE")
    if coupon:
        row = ao_next_match_row_container(coupon)
        if row:
            markets["OOE"] = add_display("OOE", ao_parse_ooe(row))
    coupon = soup.select_one(".coupon.couponHFT")
    if coupon:
        row = ao_next_match_row_container(coupon)
        if row:
            markets["HFT"] = {"display_name_zh": AO_DISPLAY_ZH["HFT"], "combos": ao_parse_hft(row)}

    for code, cls in [("FGS","couponFGS"), ("LGS","couponLGS"), ("AGS","couponAGS")]:
        coupon = soup.select_one(f".coupon.{cls}")
        if coupon:
            row = ao_next_match_row_container(coupon)
            if row:
                markets[code] = {"display_name_zh": AO_DISPLAY_ZH[code], "players": ao_parse_scorer_market(row)}

    coupon = soup.select_one(".coupon.couponMSP")
    if coupon:
        row = ao_next_match_row_container(coupon)
        if row:
            markets["MSP"] = {"display_name_zh": AO_DISPLAY_ZH["MSP"], "items": ao_parse_msp(row)}

    return match_meta, markets

class HKJCDetailedOddsScraper:
    def __init__(self, output_dir: Path = Path("hkjc/odds")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def scrape(self, event_id: str) -> Optional[Dict[str, Any]]:
        url = f"https://bet.hkjc.com/ch/football/allodds/{event_id}"
        logger.info("🌐 Scraping HKJC All Odds for event: %s", event_id)
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/118.0.5993.117 Safari/537.36"
                )
            )
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=90000)
                try:
                    await page.wait_for_selector(".match-info", timeout=30000)
                except Exception:
                    logger.warning("HKJC detailed odds: .match-info not found for %s", event_id)
                html = await page.content()
            except Exception as e:
                logger.error("Error scraping HKJC detailed odds %s: %s", event_id, e)
                await browser.close()
                return None
            await browser.close()

        try:
            match_meta, markets = ao_parse_allodds_from_html(html)
            out_data = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "event_id": event_id,
                "url": url,
                "match": match_meta,
                "markets": markets,
            }
            out_path = self.output_dir / f"hkjc_odds_{event_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
            return out_data
        except Exception as e:
            logger.error("Error parsing HKJC detailed odds %s: %s", event_id, e)
            return None

# ---------------------------------------------------------------------------
# HKJC HOME scraper (bet.hkjc.com first, speedbet fallback, clicks "顯示更多", tvChannels)
# ---------------------------------------------------------------------------
class HKJCHomeScraper:
    """
    Scrapes HKJC home pages to collect /allodds/<event_id> links and embedded ids.
    Tries bet.hkjc.com first; if zero found, tries speedbet.hkjc.com (no login).
    """
    def __init__(self, urls=None):
        if urls is None:
            urls = [
                "https://bet.hkjc.com/ch/football/home",
                "https://speedbet.hkjc.com/ch/football/home",
            ]
        self.urls = urls

    async def _click_show_more(self, page):
        xpaths = [
            "//div[contains(text(), '顯示更多')]",
            "//button[contains(text(), '顯示更多')]",
            "//span[contains(text(), '顯示更多')]",
            "//a[contains(text(), '顯示更多')]"
        ]
        for xp in xpaths:
            try:
                els = await page.query_selector_all(f"xpath={xp}")
                for el in els:
                    if await el.is_visible():
                        await el.scroll_into_view_if_needed()
                        await asyncio.sleep(0.3)
                        await el.click()
                        await asyncio.sleep(1.0)
                        return True
            except Exception:
                continue
        return False

    async def _scrape_one(self, url: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/118.0.5993.117 Safari/537.36"
                )
            )
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=90000)
                await asyncio.sleep(2.0)
                await self._click_show_more(page)
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(1.0)
                html = await page.content()
            finally:
                await browser.close()

        soup = BeautifulSoup(html, "html.parser")
        candidates = []
        # href-based
        for a in soup.find_all("a", href=True):
            m = re.search(r"/allodds/(\d+)", a["href"])
            if m:
                eid = m.group(1)
                candidates.append((eid, a.get_text(" ", strip=True), a))
        # data-attr based
        for el in soup.find_all(True):
            eid = el.get("data-eventid") or el.get("data-event-id") or el.get("data-event")
            if eid and re.match(r"^\d+$", eid):
                candidates.append((eid, el.get_text(" ", strip=True), el))
        # tvChannels class-based
        for m in re.finditer(r'class="[^"]*\btvChannels\s+(\d+)\b', html):
            eid = m.group(1)
            candidates.append((eid, "", None))

        seen = set()
        rows = []
        for eid, txt, node in candidates:
            if eid in seen:
                continue
            seen.add(eid)
            home = away = None
            text_block = txt or ""
            m_vs = re.split(r"\s+vs\s+|\s+VS\s+|\s+對\s+|\s+v\s+", text_block)
            if len(m_vs) >= 2:
                home, away = m_vs[0].strip(), m_vs[1].strip()
            rows.append({
                "event_id": eid,
                "home_team": home,
                "away_team": away,
                "raw_text": text_block
            })
        logger.info("HKJCHomeScraper %s collected %d event ids", url, len(rows))
        return rows

    async def scrape(self) -> List[Dict[str, Any]]:
        all_rows: List[Dict[str, Any]] = []
        for u in self.urls:
            rows = await self._scrape_one(u)
            all_rows.extend(rows)
            if rows:
                break
        dedup = {r["event_id"]: r for r in all_rows}
        return list(dedup.values())

# ---------------------------------------------------------------------------
# HKJC Bulk odds collector (with skip cache)
# ---------------------------------------------------------------------------
class HKJCBulkOddsCollector:
    def __init__(self, home_scraper: HKJCHomeScraper, detailed_scraper: HKJCDetailedOddsScraper,
                 existing_cache: Optional[Dict[str, Any]] = None, skip_ids: Optional[set] = None):
        self.home_scraper = home_scraper
        self.detailed_scraper = detailed_scraper
        self.existing_cache = existing_cache or {}
        self.skip_ids = set(skip_ids or [])

    async def collect(self, max_events: int = 200, concurrency: int = 4) -> Tuple[Dict[str, Dict[str, Any]], set, List[Dict[str, Any]]]:
        home_rows = await self.home_scraper.scrape()
        event_ids = [r["event_id"] for r in home_rows if r["event_id"] not in self.skip_ids and r["event_id"] not in self.existing_cache][:max_events]
        logger.info("Bulk HKJC odds: %d event_ids to scrape (capped to %d, skipped %d cached)", len(event_ids), max_events, len(self.skip_ids) + len(self.existing_cache))
        sem = asyncio.Semaphore(concurrency)
        results: Dict[str, Dict[str, Any]] = dict(self.existing_cache)
        failed: List[str] = []

        async def worker(eid: str):
            nonlocal results, failed
            async with sem:
                try:
                    data = await self.detailed_scraper.scrape(eid)
                    if data:
                        results[eid] = data
                        self.skip_ids.add(eid)
                except Exception as e:
                    logger.warning("Bulk HKJC odds failed for %s: %s", eid, e)
                    failed.append(eid)

        await asyncio.gather(*(worker(eid) for eid in event_ids))
        out_dir = self.detailed_scraper.output_dir
        out_path = out_dir / f"hkjc_allodds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        payload = {
            "metadata": {
                "scraped_at": datetime.now().isoformat(),
                "source": "HKJC",
                "total_event_ids": len(event_ids),
                "succeeded": len(results),
                "failed": failed,
            },
            "events": list(results.values())
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("💾 Saved HKJC all odds to: %s (events=%d)", out_path, len(results))
        return results, self.skip_ids, home_rows

# ---------------------------------------------------------------------------
# Titan007 stats scraper (restored)
# ---------------------------------------------------------------------------
async def scrape_match_stats_from_analysis_page(titan_match_id: str) -> Dict[str, Any]:
    url = f"https://zq.titan007.com/analysis/{titan_match_id}.htm"
    logger.info("🔍 Scraping analysis stats for Titan match ID: %s", titan_match_id)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until='networkidle', timeout=60000)
            await asyncio.sleep(3.0)
            try:
                await page.wait_for_selector("table", timeout=3000)
            except Exception:
                pass
            content = await page.content()
            if DEBUG_INSTRUMENTATION_AVAILABLE:
                try:
                    save_rendered_html("titan_analysis", titan_match_id, content)
                except Exception:
                    pass
            soup = BeautifulSoup(content, "html.parser")
            data = {
                "match_id": titan_match_id,
                "scraped_at": datetime.now().isoformat(),
                "url": url,
                "sections": {},
                "stats_available": False
            }

            page_text = soup.get_text(separator=' ', strip=True)
            no_data_patterns = ["暫無數據", "數據統計中", "未開始", "資料準備中", "尚無相關資料", "No data"]
            for pat in no_data_patterns:
                if pat in page_text:
                    logger.warning("⚠️ Titan %s: '%s' found — no stats available", titan_match_id, pat)
                    data["error"] = f"No stats: {pat}"
                    if DEBUG_INSTRUMENTATION_AVAILABLE:
                        try:
                            save_parsed_json("titan_analysis_parsed", titan_match_id, data)
                        except Exception:
                            pass
                    return data

            def extract_section_by_regex(regex: str) -> Optional[Any]:
                header = soup.find(string=re.compile(regex))
                if not header:
                    return None
                parent = None
                try:
                    if isinstance(header, Tag):
                        parent = header.find_parent()
                    else:
                        parent = header.parent if hasattr(header, "parent") else None
                except Exception:
                    parent = header.parent if hasattr(header, "parent") else None
                if not parent:
                    parent = header.parent if hasattr(header, "parent") else None
                if not parent:
                    return None
                tables = parent.find_all('table')
                for table in tables:
                    parsed = extract_table_data(table)
                    if parsed:
                        return parsed
                txt = parent.get_text(separator=' | ', strip=True)
                if txt and len(txt) > 30:
                    return [{"text_content": txt}]
                return None

            sections_to_try = [
                ("league_standings", r'聯賽積分排名'),
                ("head_to_head", r'對賽往績'),
                ("data_comparison", r'數據對比'),
                ("referee_stats", r'裁判統計'),
                ("league_trend", r'聯賽盤路走勢'),
                ("same_trend", r'相同盤路'),
                ("goal_distribution", r'入球數/上下半場入球分布'),
                ("halftime_fulltime", r'半全場'),
                ("goal_count", r'進球數/單雙'),
                ("goal_time", r'進球時間'),
                ("future_matches", r'未來五場'),
                ("pre_match_brief", r'賽前簡報'),
                ("season_stats_comparison", r'本賽季數據統計比較'),
            ]

            sections_found = 0
            for key, regex in sections_to_try:
                try:
                    sec = extract_section_by_regex(regex)
                    if sec:
                        data["sections"][key] = sec
                        sections_found += 1
                        logger.debug("✅ Extracted section %s for match %s", key, titan_match_id)
                except Exception:
                    logger.debug("Failed extracting section %s", key)

            formation_header = soup.find(string=re.compile(r'陣容情況'))
            if formation_header:
                parent = formation_header.find_parent() if hasattr(formation_header, "find_parent") else formation_header.parent
                if parent:
                    data["sections"]["team_formation"] = parent.get_text(separator=' | ', strip=True)
                    sections_found += 1

            try:
                home_avg, home_list = extract_ratings_or_average_from_text(page_text)
                away_avg, away_list = None, []
                m_home = re.search(r'主隊近10場平均評分[:：]?\s*([0-9\.\s]{5,200})', page_text)
                m_away = re.search(r'客隊近10場平均評分[:：]?\s*([0-9\.\s]{5,200})', page_text)
                if m_home:
                    home_avg, home_list = extract_ratings_or_average_from_text("主隊近10場平均評分:" + m_home.group(1))
                if m_away:
                    away_avg, away_list = extract_ratings_or_average_from_text("客隊近10場平均評分:" + m_away.group(1))
                if away_avg is None:
                    away_avg, away_list = extract_ratings_or_average_from_text(page_text)
                if home_avg is not None:
                    data["home_rating"] = home_avg
                    data["home_recent_ratings"] = home_list
                if away_avg is not None:
                    data["away_rating"] = away_avg
                    data["away_recent_ratings"] = away_list
            except Exception as e:
                logger.debug("Rating extraction exception: %s", e)

            if sections_found >= 1 or data.get("home_rating") or data.get("away_rating"):
                data["stats_available"] = True
                logger.info("✅ Titan %s: scraped %d sections", titan_match_id, sections_found)
            else:
                data["error"] = f"Insufficient stats ({sections_found} sections)"
                logger.warning("⚠️ Titan %s: insufficient stats (%d sections)", titan_match_id, sections_found)

            if DEBUG_INSTRUMENTATION_AVAILABLE:
                try:
                    save_parsed_json("titan_analysis_parsed", titan_match_id, data)
                except Exception:
                    pass
            return data
        except Exception as e:
            logger.exception("Error scraping analysis stats for Titan match %s: %s", titan_match_id, e)
            return {
                "match_id": titan_match_id,
                "scraped_at": datetime.now().isoformat(),
                "url": url,
                "stats_available": False,
                "error": str(e)
            }
        finally:
            try:
                await browser.close()
            except Exception:
                logger.debug("Browser close failed (ignored) for %s", titan_match_id)

# ---------------------------------------------------------------------------
# Titan table helpers
# ---------------------------------------------------------------------------
def extract_table_data_from_real_table(table_elem: Tag) -> List[Dict[str, str]]:
    rows = table_elem.find_all('tr')
    if not rows:
        return []
    header_row = None
    for row in rows:
        cells = row.find_all(['th', 'td'])
        if cells and 2 <= len(cells) <= 30:
            cell_texts = [c.get_text(strip=True) for c in cells]
            header_words = ['賽', '勝', '平', '負', '得', '失', '積分', '勝率', '排名', '主場', '客場']
            if any(any(w in txt for w in header_words) for txt in cell_texts):
                header_row = row
                break
    if not header_row:
        header_row = rows[0]
    headers = []
    for cell in header_row.find_all(['th', 'td']):
        text = cell.get_text(strip=True)
        clean_text = re.sub(r'\s+', ' ', text).strip() if text else f"col_{len(headers)}"
        headers.append(clean_text)
    if len(headers) < 2 or len(headers) > 30:
        return []
    data = []
    start_index = rows.index(header_row) + 1 if header_row in rows else 1
    for row in rows[start_index:]:
        cells = row.find_all(['td', 'th'])
        if not cells or len(cells) < 2:
            continue
        row_data = {}
        for i, cell in enumerate(cells):
            if i >= len(headers):
                break
            cell_text = cell.get_text(strip=True)
            clean_text = re.sub(r'\s+', ' ', cell_text).strip()
            if clean_text:
                row_data[headers[i]] = clean_text
                link = cell.find('a')
                if link and link.get('href'):
                    row_data[f"{headers[i]}_link"] = link.get('href')
        if row_data and len(row_data) >= 2:
            data.append(row_data)
    return data


def extract_table_data(table_elem: Tag) -> List[Dict[str, str]]:
    rows = table_elem.find_all('tr')
    if not rows:
        return []
    first_row_cells = rows[0].find_all(['th', 'td'])
    if len(first_row_cells) > 50:
        nested_tables = table_elem.find_all('table', class_=re.compile(r'oddsTable|dataTable|statsTable', re.I))
        if nested_tables:
            return extract_table_data_from_real_table(nested_tables[0])
        for nested in table_elem.find_all('table'):
            nested_rows = nested.find_all('tr')
            if nested_rows:
                first_nested_cells = nested_rows[0].find_all(['th', 'td'])
                if 3 <= len(first_nested_cells) <= 20:
                    return extract_table_data_from_real_table(nested)
        return []
    return extract_table_data_from_real_table(table_elem)

# ---------------------------------------------------------------------------
# MacauSlot odds scraper
# ---------------------------------------------------------------------------
class MacauSlotOddsScraper:
    def __init__(self):
        self.base_url = "https://www.macau-slot.com/content/soccer/coming_bet.html"
        self.output_dir = Path("macauslot/odds")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def _scrape_page_data_js(self, page) -> List[Dict]:
        try:
            return await page.evaluate("""() => {
                const matches = [];
                const containers = document.querySelectorAll('li.msl-ls-item, li.msl-odds-tr');

                containers.forEach(container => {
                    const eventId = container.getAttribute('data-ev-id');
                    if (!eventId) return;

                    const timeElem = container.querySelector('.minute');
                    const homeTeamElem = container.querySelector('.msl-odd-td-host');
                    const awayTeamElem = container.querySelector('.msl-odd-td-guest');
                    const flagWrap = container.querySelector('.msl-flag-wrap');

                    const home = homeTeamElem ? homeTeamElem.textContent.trim() : '';
                    const away = awayTeamElem ? awayTeamElem.textContent.trim() : '';
                    if (!home || !away) return;

                    const match = {
                        event_id: eventId,
                        time: timeElem ? timeElem.textContent.trim() : '',
                        competition: flagWrap ? (flagWrap.getAttribute('data-original-title') || '').trim() : '',
                        competition_short: flagWrap ? ((flagWrap.querySelector('.short') || {}).textContent || '').trim() : '',
                        home_team: home,
                        away_team: away,
                        odds: {
                            asian_handicap: [],
                            over_under: [],
                            home_draw_away: { home_odds: null, draw_odds: null, away_odds: null }
                        }
                    };

                    const oddsWrapper = container.querySelector('.msl-cm-odds-wrapper');
                    if (!oddsWrapper) {
                        matches.push(match);
                        return;
                    }

                    const stdCol = oddsWrapper.querySelector('.msl-odds-td.col-3.msl-odd-btn-bets') ||
                                   oddsWrapper.querySelector('.msl-odds-td.col-3');
                    if (stdCol) {
                        const buttons = stdCol.querySelectorAll('button.msl-bet');
                        buttons.forEach(btn => {
                            const sideBadge = btn.querySelector('.badge_front');
                            const oddsBadge = btn.querySelector('.badge');
                            if (!sideBadge || !oddsBadge) return;

                            const side = sideBadge.textContent.trim();
                            const oddsText = oddsBadge.textContent.trim();
                            const m = oddsText.match(/[\\d.]+/);
                            const odds = m ? parseFloat(m[0]) : null;
                            if (!odds) return;

                            if (side === '主') match.odds.home_draw_away.home_odds = odds;
                            else if (side === '和') match.odds.home_draw_away.draw_odds = odds;
                            else if (side === '客') match.odds.home_draw_away.away_odds = odds;
                        });
                    }

                    const ahSections = oddsWrapper.querySelectorAll(
                        '.msl-odds-td.col-1, .msl-odds-td.msl-odd-td-oddstype.col-1'
                    );
                    ahSections.forEach(section => {
                        const buttons = section.querySelectorAll('button.msl-bet');
                        buttons.forEach(btn => {
                            const sideBadge = btn.querySelector('.badge_left');
                            const lineBadge = btn.querySelector('.badge_front');
                            const oddsBadge = btn.querySelector('.badge');
                            if (!sideBadge || !lineBadge || !oddsBadge) return;

                            const side = sideBadge.textContent.trim();
                            const line = lineBadge.textContent.trim();
                            const oddsText = oddsBadge.textContent.trim();
                            const m = oddsText.match(/[\\d.]+/);
                            const odds = m ? parseFloat(m[0]) : null;
                            if (!odds) return;

                            let entry = match.odds.asian_handicap.find(x => x.handicap_value === line);
                            if (!entry) {
                                entry = { handicap_value: line, home_odds: null, away_odds: null };
                                match.odds.asian_handicap.push(entry);
                            }
                            if (side === '主') entry.home_odds = odds;
                            else if (side === '客') entry.away_odds = odds;
                        });
                    });

                    const ouSections = oddsWrapper.querySelectorAll(
                        '.msl-odds-td.col-2, .msl-odds-td.msl-odd-td-oddstype.col-2'
                    );
                    ouSections.forEach(section => {
                        const buttons = section.querySelectorAll('button.msl-bet');
                        buttons.forEach(btn => {
                            const sideBadge = btn.querySelector('.badge_left');
                            const lineBadge = btn.querySelector('.badge_front');
                            const oddsBadge = btn.querySelector('.badge');
                            if (!sideBadge || !lineBadge || !oddsBadge) return;

                            const side = sideBadge.textContent.trim();
                            const line = lineBadge.textContent.trim();
                            const oddsText = oddsBadge.textContent.trim();
                            const m = oddsText.match(/[\\d.]+/);
                            const odds = m ? parseFloat(m[0]) : null;
                            if (!odds) return;

                            let entry = match.odds.over_under.find(x => x.goal_line === line);
                            if (!entry) {
                                entry = { goal_line: line, over_odds: null, under_odds: null };
                                match.odds.over_under.push(entry);
                            }
                            if (side === '上') entry.over_odds = odds;
                            else if (side === '下') entry.under_odds = odds;
                        });
                    });

                    matches.push(match);
                });

                return matches;
            }""")
        except Exception as e:
            logger.error("⚠️ JS scrape failed: %s", e)
            return []

    async def scrape(self, max_pages: int = 20) -> List[Dict]:
        logger.info("🌐 Scraping Macau Slot live odds...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
            page = await context.new_page()
            try:
                goto_retries = 3
                for attempt in range(1, goto_retries + 1):
                    try:
                        await page.goto(self.base_url, wait_until="domcontentloaded", timeout=30000)
                        break
                    except Exception as e:
                        logger.error("Macau page.goto failed (attempt %d/%d): %s", attempt, goto_retries, e)
                        if attempt == goto_retries:
                            raise
                        await asyncio.sleep(1.5 * attempt)

                await asyncio.sleep(2)
                if DEBUG_INSTRUMENTATION_AVAILABLE:
                    try:
                        content_initial = await page.content()
                        save_rendered_html("macau_page", "initial", content_initial)
                    except Exception:
                        pass
                try:
                    select = await page.query_selector('select.msl-cm-pager[name="msl_record_per_page"]')
                    if select:
                        await select.select_option('50')
                        await asyncio.sleep(1)
                except Exception:
                    pass
                all_matches = []
                for page_num in range(1, max_pages + 1):
                    if page_num > 1:
                        btn = await page.query_selector(f'input.msl-menu-page[value="{page_num}"]')
                        if btn and await btn.is_visible():
                            await btn.click()
                            await asyncio.sleep(1.5)
                        else:
                            break
                    if DEBUG_INSTRUMENTATION_AVAILABLE:
                        try:
                            content_page = await page.content()
                            save_rendered_html("macau_page", f"page_{page_num}", content_page)
                        except Exception:
                            pass
                    page_data = await self._scrape_page_data_js(page)
                    if page_data:
                        all_matches.extend(page_data)
                        logger.info("Page %d: Found %d matches", page_num, len(page_data))
                        if DEBUG_INSTRUMENTATION_AVAILABLE:
                            try:
                                save_parsed_json("macau_matches", f"page{page_num}", {"matches": page_data})
                            except Exception:
                                pass
                    else:
                        break
                await browser.close()
                return all_matches
            except Exception as e:
                logger.exception("❌ Macau scrape error: %s", e)
                await browser.close()
                return []

    def save_to_json(self, data: List[Dict], filename: Optional[str] = None) -> str:
        if not filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"macauslot_odds_{ts}.json"
        else:
            filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "metadata": {
                "scraped_at": datetime.now().isoformat(),
                "source": "MacauSlot",
                "url": self.base_url,
                "total_matches": len(data)
            },
            "matches": data
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logger.info("💾 Saved Macau odds to: %s", filename)
        return str(filename)

# ---------------------------------------------------------------------------
# LiveMatchMatcher
# ---------------------------------------------------------------------------
class LiveMatchMatcher:
    def __init__(self, min_similarity_threshold: float = 0.75, time_tolerance_minutes: int = 30,
                 prioritize_similarity: bool = True):
        self.matched_games: List[Dict[str, Any]] = []
        self.unmatched_games: List[Dict[str, Any]] = []
        self.min_similarity_threshold = min_similarity_threshold
        self.time_tolerance_minutes = time_tolerance_minutes
        self.prioritize_similarity = prioritize_similarity
        self.data_quality_metrics = {
            "total_hkjc_matches": 0,
            "total_titan_matches": 0,
            "potential_matches_checked": 0,
            "high_confidence_matches": 0,
            "low_confidence_matches": 0
        }
        self.raw_hkjc_matches = []
        self.raw_titan_matches = []
        self.macau_mapping = {}
        self.ai_cache = load_ai_cache()
        self.hkjc_bulk_odds: Dict[str, Dict[str, Any]] = {}
        self.hkjc_odds_processed: set = load_cache_set(HKJC_ODDS_PROCESSED_PATH)
        self.titan_stats_processed: set = load_cache_set(TITAN_STATS_PROCESSED_PATH)

    def calculate_name_similarity(self, name1: str, name2: str) -> float:
        def clean_name(name: str) -> str:
            if not name:
                return ""
            name = re.sub(r'\[\d+\]', '', name)
            name = re.sub(r'\(中\)', '', name)
            name = re.sub(r'(女足|女子)$', '', name)
            return name.strip().lower()
        clean1 = clean_name(name1)
        clean2 = clean_name(name2)
        if not clean1 or not clean2:
            return 0.0
        return SequenceMatcher(None, clean1, clean2).ratio()

    def normalize_time(self, time_str: str) -> Optional[datetime]:
        if not time_str:
            return None
        s = time_str.strip()
        now = datetime.now()
        formats = [
            "%d/%m/%Y %H:%M",
            "%d/%m %H:%M",
            "%m/%d/%Y %H:%M",
            "%m/%d %H:%M",
            "%Y-%m-%d %H:%M",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(s, fmt)
                if fmt in ("%d/%m %H:%M", "%m/%d %H:%M"):
                    dt = dt.replace(year=now.year)
                return dt
            except Exception:
                continue
        if re.match(r'^\d{1,2}:\d{2}$', s):
            try:
                h, m = map(int, s.split(":"))
                return datetime(now.year, now.month, now.day, h, m)
            except Exception:
                pass
        try:
            return datetime.fromisoformat(s)
        except Exception:
            logger.debug("normalize_time failed for %s", time_str)
            return None

    def is_exact_time_match(self, time1: Optional[datetime], time2: Optional[datetime]) -> bool:
        if not time1 or not time2:
            return False
        diff = abs((time1 - time2).total_seconds() / 60)
        return diff <= self.time_tolerance_minutes

    def are_teams_similar_enough(self, hkjc_home: str, hkjc_away: str,
                                 titan_home: str, titan_away: str) -> Tuple[bool, float, bool]:
        home_sim = self.calculate_name_similarity(hkjc_home, titan_home)
        away_sim = self.calculate_name_similarity(hkjc_away, titan_away)
        home_swapped = self.calculate_name_similarity(hkjc_home, titan_away)
        away_swapped = self.calculate_name_similarity(hkjc_away, titan_home)
        if home_sim >= self.min_similarity_threshold or away_sim >= self.min_similarity_threshold:
            return True, (home_sim + away_sim) / 2, False
        if home_swapped >= self.min_similarity_threshold or away_swapped >= self.min_similarity_threshold:
            return True, (home_swapped + away_swapped) / 2, True
        return False, 0.0, False

    def validate_match_data(self, match_data: Dict) -> bool:
        required_fields = ['home_team', 'away_team']
        for f in required_fields:
            if not match_data.get(f) or len(str(match_data[f]).strip()) < 2:
                return False
        for team_field in ['home_team', 'away_team']:
            name = match_data.get(team_field, "")
            if name and len(re.sub(r'[^a-zA-Z\u4e00-\u9fff]', '', name)) == 0:
                return False
        return True

    def filter_future_hkjc_matches(self, hkjc_matches: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        now = datetime.now()
        future, started = [], []
        for m in hkjc_matches:
            nt = m.get("normalized_time")
            if isinstance(nt, str):
                try:
                    nt = datetime.fromisoformat(nt)
                except Exception:
                    nt = None
            if nt and now >= nt:
                started.append(m)
            else:
                future.append(m)
        if started:
            logger.info("Filtered out %d started HKJC matches", len(started))
        return future, started

    def enrich_hkjc_with_home_event_ids(self, hkjc_matches: List[Dict], home_rows: List[Dict]):
        if not home_rows:
            return
        for m in hkjc_matches:
            if m.get("event_id"):
                continue
            best = None
            best_sim = 0.0
            for sb in home_rows:
                home, away = sb.get("home_team"), sb.get("away_team")
                if not home or not away:
                    continue
                teams_similar, avg_sim, _ = self.are_teams_similar_enough(
                    m.get("home_team", ""), m.get("away_team", ""), home, away
                )
                if teams_similar and avg_sim > best_sim:
                    best_sim = avg_sim
                    best = sb
            if best and best_sim >= 0.70:
                m["event_id"] = best.get("event_id")
                logger.info("Attached event_id %s to HKJC %s vs %s (sim=%.2f)",
                            m["event_id"], m.get("home_team"), m.get("away_team"), best_sim)

    # ----------------------- HKJC scraper -----------------------
    async def scrape_hkjc_matches(self) -> List[Dict]:
        matches: List[Dict] = []
        raw_matches: List[Dict] = []
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                cprint("🌐 Loading HKJC matches live...", Fore.BLUE)
                await page.goto("https://bet.hkjc.com/ch/football/had", wait_until='domcontentloaded', timeout=60000)
                await asyncio.sleep(2)
                try:
                    content = await page.content()
                    if DEBUG_INSTRUMENTATION_AVAILABLE:
                        save_rendered_html("hkjc_page", "index", content)
                except Exception:
                    pass
                await self.click_show_more_hkjc(page)
                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')
                match_rows = soup.find_all('div', class_='match-row') or soup.find_all('div', class_='event-row')
                cprint(f"🔍 Found {len(match_rows)} match rows on HKJC", Fore.CYAN)
                for i, row in enumerate(match_rows):
                    try:
                        match_data = await self.extract_hkjc_match_data(row)
                        if match_data and self.validate_match_data(match_data):
                            norm_time_dt = self.normalize_time(match_data.get('date', ''))
                            norm_time_str = norm_time_dt.isoformat() if norm_time_dt else None
                            raw = {
                                "source": "HKJC",
                                "match_id": match_data.get('match_id', ''),
                                "event_id": match_data.get('event_id', ''),
                                "home_team": match_data['home_team'],
                                "away_team": match_data['away_team'],
                                "match_time_original": match_data.get('date', ''),
                                "normalized_time": norm_time_str,
                                "normalized_time_str": norm_time_str,
                                "tournament": match_data.get('tournament', ''),
                                "scraped_at": datetime.now().isoformat()
                            }
                            raw_matches.append(raw)
                            matches.append(raw)
                            if i < 3:
                                cprint(f"  Sample: {raw['home_team']} vs {raw['away_team']} (event_id={raw.get('event_id','')})", Fore.MAGENTA)
                    except Exception as e:
                        if i < 3:
                            cprint(f"  ⚠️ Error in HKJC row {i+1}: {e}", Fore.YELLOW)
                        continue
                self.data_quality_metrics['total_hkjc_matches'] = len(matches)
                self.raw_hkjc_matches = raw_matches
                if DEBUG_INSTRUMENTATION_AVAILABLE:
                    try:
                        save_parsed_json("hkjc_index_parsed", "index", {"raw_matches": raw_matches, "count": len(raw_matches)})
                    except Exception:
                        pass
                cprint(f"✅ Successfully extracted {len(matches)} HKJC matches", Fore.GREEN)
                return matches
            except Exception as e:
                cprint(f"❌ Error scraping HKJC: {e}", Fore.RED)
                return []
            finally:
                await browser.close()

    async def click_show_more_hkjc(self, page):
        try:
            xpaths = [
                "//div[contains(text(), '顯示更多')]",
                "//button[contains(text(), '顯示更多')]",
                "//span[contains(text(), '顯示更多')]",
                "//a[contains(text(), '顯示更多')]"
            ]
            for xp in xpaths:
                elements = await page.query_selector_all(f"xpath={xp}")
                for el in elements:
                    try:
                        if await el.is_visible():
                            await el.scroll_into_view_if_needed()
                            await asyncio.sleep(0.5)
                            await el.click()
                            cprint("  ✅ Clicked 'Show More' for HKJC", Fore.BLUE)
                            await asyncio.sleep(1.5)
                            return True
                    except Exception:
                        continue
            return False
        except Exception as e:
            cprint(f"  ⚠️ Could not find 'Show More' ({e})", Fore.YELLOW)
            return False

    async def extract_hkjc_match_data(self, match_row) -> Optional[Dict]:
        try:
            match_id_elem = match_row.find('div', class_='fb-id')
            match_id = match_id_elem.get_text(strip=True) if match_id_elem else None
            date_elem = match_row.find('div', class_='date')
            date = date_elem.get_text(strip=True) if date_elem else ""
            tourn_elem = match_row.find('div', class_='tourn')
            tournament = ""
            if tourn_elem and tourn_elem.find('img'):
                tournament = tourn_elem.find('img').get('title', '') or ""
            home_team, away_team = self.extract_hkjc_teams(match_row)

            event_id = None
            link = match_row.find('a', href=True)
            if link and link['href']:
                m = re.search(r'/allodds/(\d+)', link['href'])
                if m:
                    event_id = m.group(1)

            if not home_team or not away_team:
                return None
            return {'match_id': match_id, 'event_id': event_id, 'date': date, 'tournament': tournament, 'home_team': home_team, 'away_team': away_team}
        except Exception as e:
            logger.debug("extract_hkjc_match_data error: %s", e)
            return None

    def extract_hkjc_teams(self, match_row) -> Tuple[str, str]:
        home_team = ""
        away_team = ""
        try:
            team_icon = match_row.find('div', class_='teamIconSmall')
            if team_icon:
                team_container = team_icon.find('div', title=True)
                if team_container:
                    divs = team_container.find_all('div')
                    if len(divs) >= 2:
                        home_team = divs[0].get_text(strip=True)
                        away_team = divs[1].get_text(strip=True)
        except Exception as e:
            logger.debug("extract_hkjc_teams error: %s", e)
        return home_team, away_team

    # ----------------------- Titan007 scraper -----------------------
    async def scrape_titan007_matches(self) -> List[Dict]:
        matches = []
        raw_matches = []
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                cprint("🌐 Loading Titan007 matches live...", Fore.BLUE)
                await page.goto("https://live.titan007.com/indexall_big.aspx", wait_until='networkidle', timeout=30000)
                await asyncio.sleep(1.5)
                try:
                    content = await page.content()
                    if DEBUG_INSTRUMENTATION_AVAILABLE:
                        save_rendered_html("titan_index", "index", content)
                    else:
                        _ = content
                except Exception:
                    pass
                soup = BeautifulSoup(content, 'html.parser')
                main_table = None
                for table in soup.find_all('table'):
                    txt = table.get_text()
                    if '時間' in txt and '比賽球隊' in txt:
                        main_table = table
                        break
                if not main_table:
                    cprint("❌ Could not find main Titan007 table", Fore.RED)
                    return []
                rows = main_table.find_all('tr')
                cprint(f"🔍 Found {len(rows)} rows in Titan007 table", Fore.CYAN)
                time_col_idx = 1
                status_col_idx = 2
                for i, row in enumerate(rows):
                    try:
                        if not row.get_text(strip=True):
                            continue
                        if '時間' in row.get_text() and '比賽球隊' in row.get_text():
                            cells = row.find_all(['td', 'th'])
                            for idx, cell in enumerate(cells):
                                txt = cell.get_text(strip=True)
                                if txt == '時間':
                                    time_col_idx = idx
                                elif txt == '狀態':
                                    status_col_idx = idx
                            continue
                        team1 = row.find('a', id=lambda x: x and x.startswith('team1_'))
                        team2 = row.find('a', id=lambda x: x and x.startswith('team2_'))
                        if not team1 or not team2:
                            continue
                        match_id = team1.get('id', '').replace('team1_', '')
                        league = "Unknown"
                        cells = row.find_all(['td', 'th'])
                        if cells:
                            league_text = cells[0].get_text(strip=True)
                            if league_text and league_text != '時間' and '比賽' not in league_text:
                                league = league_text
                        scheduled_time = ""
                        if len(cells) > time_col_idx:
                            scheduled_time = cells[time_col_idx].get_text(strip=True)
                        status = ""
                        if len(cells) > status_col_idx:
                            status = cells[status_col_idx].get_text(strip=True)
                        home_team = re.sub(r'\[\d+\]|\(中\)', '', team1.get_text(strip=True)).strip()
                        away_team = re.sub(r'\[\d+\]|\(中\)', '', team2.get_text(strip=True)).strip()
                        if not home_team or not away_team or len(home_team) < 2 or len(away_team) < 2:
                            continue
                        score = ""
                        for cell in cells:
                            cell_text = cell.get_text(strip=True)
                            if '-' in cell_text and len(cell_text) <= 7:
                                score = cell_text
                                break
                        normalized_time = None
                        if scheduled_time and re.match(r'^\d{1,2}:\d{2}$', scheduled_time):
                            try:
                                hour, minute = map(int, scheduled_time.split(":"))
                                today = datetime.now()
                                normalized_time = datetime(today.year, today.month, today.day, hour, minute)
                            except Exception:
                                normalized_time = None
                        raw_match = {
                            "source": "Titan007",
                            "match_id": match_id,
                            "league": league,
                            "home_team": home_team,
                            "away_team": away_team,
                            "scheduled_time_original": scheduled_time,
                            "match_status": status,
                            "score": score,
                            "normalized_time": normalized_time,
                            "normalized_time_str": normalized_time.isoformat() if normalized_time else None,
                            "scraped_at": datetime.now().isoformat()
                        }
                        raw_matches.append(raw_match)
                        if self.validate_match_data(raw_match):
                            matches.append(raw_match)
                        if len(matches) <= 3:
                            cprint(f"  Sample: {home_team} vs {away_team}", Fore.MAGENTA)
                    except Exception as e:
                        if i < 5:
                            cprint(f"  ⚠️ Error parsing Titan row {i + 1}: {e}", Fore.YELLOW)
                        continue
                self.data_quality_metrics['total_titan_matches'] = len(matches)
                self.raw_titan_matches = raw_matches
                if DEBUG_INSTRUMENTATION_AVAILABLE:
                    try:
                        save_parsed_json("titan_index_parsed", "index", {"raw_matches": raw_matches, "matches": matches})
                    except Exception:
                        pass
                cprint(f"✅ Successfully extracted {len(matches)} Titan007 matches", Fore.GREEN)
                return matches
            except Exception as e:
                cprint(f"❌ Error scraping Titan007: {e}", Fore.RED)
                return []
            finally:
                await browser.close()

    # ----------------------- Matching orchestration -----------------------
    async def find_matching_games(self) -> Tuple[List[Dict], List[Dict]]:
        cprint("\n" + "=" * 80, Fore.WHITE)
        cprint("🔍 FINDING MATCHES (HKJC + Titan007 + Macau Slot)", Fore.WHITE)
        cprint("=" * 80, Fore.WHITE)

        if DEBUG_INSTRUMENTATION_AVAILABLE:
            try:
                init_debug_session()
                log_info("Session started", {
                    "min_similarity_threshold": self.min_similarity_threshold,
                    "time_tolerance_minutes": self.time_tolerance_minutes
                })
            except Exception:
                pass

        # Step 0: HKJC bulk odds FIRST
        cprint("\n📥 Step 0: Scraping HKJC All Odds (bulk)...", Fore.BLUE)
        home_scraper = HKJCHomeScraper()
        hkjc_detailed_scraper = HKJCDetailedOddsScraper()
        bulk_collector = HKJCBulkOddsCollector(
            home_scraper,
            hkjc_detailed_scraper,
            existing_cache=self.hkjc_bulk_odds,
            skip_ids=self.hkjc_odds_processed,
        )
        self.hkjc_bulk_odds, self.hkjc_odds_processed, home_rows_from_step0 = await bulk_collector.collect(max_events=200, concurrency=5)
        save_cache_set(HKJC_ODDS_PROCESSED_PATH, self.hkjc_odds_processed)
        cprint(f"✅ HKJC bulk odds collected: {len(self.hkjc_bulk_odds)} events", Fore.GREEN)

        # Step 1: Macau odds
        cprint("\n📥 Step 1: Scraping Macau Slot odds...", Fore.BLUE)
        macau_scraper = MacauSlotOddsScraper()
        macau_odds = await macau_scraper.scrape(max_pages=20)
        if macau_odds:
            macau_file = macau_scraper.save_to_json(macau_odds)
            cprint(f"✅ Macau Slot: {len(macau_odds)} matches saved to {macau_file}", Fore.GREEN)
            if DEBUG_INSTRUMENTATION_AVAILABLE:
                try:
                    save_parsed_json("macau_all", "all", {"matches": macau_odds})
                except Exception:
                    pass
        else:
            cprint("⚠️ No Macau Slot odds scraped", Fore.YELLOW)
            macau_odds = []

        # Step 2: HKJC list
        cprint("\n📥 Step 2: Scraping HKJC HAD list...", Fore.BLUE)
        hkjc_matches = await self.scrape_hkjc_matches()

        # Step 2a: Enrich HKJC event IDs using Step0 home rows (no extra scrape)
        cprint("\n📥 Step 2a: Enrich HKJC event IDs using Step0 home rows...", Fore.BLUE)
        home_rows = home_rows_from_step0
        self.enrich_hkjc_with_home_event_ids(hkjc_matches, home_rows)

        # Step 2b: Titan list
        titan_matches = await self.scrape_titan007_matches()
        cprint(f"\n📊 Match Counts:", Fore.CYAN)
        cprint(f"  HKJC: {len(hkjc_matches)} matches", Fore.CYAN)
        cprint(f"  Titan007: {len(titan_matches)} matches", Fore.CYAN)
        cprint(f"  Macau Slot: {len(macau_odds)} matches", Fore.CYAN)

        # Filter out started HKJC matches
        future_hkjc_matches, started_hkjc_matches = self.filter_future_hkjc_matches(hkjc_matches)
        if started_hkjc_matches:
            cprint(f"⏭️ Skipping {len(started_hkjc_matches)} HKJC matches that already started.", Fore.YELLOW)
        hkjc_matches = future_hkjc_matches

        if not hkjc_matches or not titan_matches:
            cprint("❌ Cannot proceed: One or both sites returned no matches", Fore.RED)
            return [], []

        # Step 3: Build Macau mapping to Titan
        cprint("\n🔄 Step 3: Building Macau odds mapping to Titan...", Fore.BLUE)
        self.macau_mapping = {}
        for titan in titan_matches:
            titan_time = titan.get("normalized_time")
            for macau in macau_odds:
                macau_time = self.normalize_time(macau.get('time', ''))
                teams_similar, avg_sim, _ = self.are_teams_similar_enough(
                    titan['home_team'], titan['away_team'],
                    macau.get('home_team', ''), macau.get('away_team', '')
                )
                time_match = self.is_exact_time_match(titan_time, macau_time)
                if teams_similar and avg_sim >= 0.70 and time_match:
                    self.macau_mapping[titan['match_id']] = macau
                if DEBUG_INSTRUMENTATION_AVAILABLE:
                    try:
                        log_mapping_decision("macau_to_titan_attempt", {
                            "titan_id": titan.get("match_id"),
                            "macau_event_id": macau.get("event_id"),
                            "avg_sim": avg_sim,
                            "time_match": time_match
                        })
                    except Exception:
                        pass
        cprint(f"  Mapped {len(self.macau_mapping)} Titan matches to Macau odds", Fore.GREEN)

        # Step 4: Match HKJC <-> Titan (name-only)
        cprint("\n🔍 Step 4: Finding matches and running AI per-match...", Fore.BLUE)
        matched = []
        unmatched_hkjc = []
        ai_rows_for_excel = []

        def classify(hkjc_match, titan_match, macau_match):
            has_h = hkjc_match is not None
            has_t = titan_match is not None
            has_m = macau_match is not None
            if has_h and has_t and has_m:
                return "hkjc_titan_macau"
            if has_h and has_t and not has_m:
                return "hkjc_titan"
            if has_h and not has_t and has_m:
                return "hkjc_macau"
            if has_h and not has_t and not has_m:
                return "hkjc_only"
            if has_t and has_m:
                return "titan_macau"
            if has_t:
                return "titan_only"
            if has_m:
                return "macau_only"
            return "unknown"

        for hkjc in hkjc_matches:
            best_match = None
            best_score = 0.0
            best_is_swapped = False

            for titan in titan_matches:
                self.data_quality_metrics['potential_matches_checked'] += 1
                teams_similar, avg_sim, is_swapped = self.are_teams_similar_enough(
                    hkjc['home_team'], hkjc['away_team'],
                    titan['home_team'], titan['away_team']
                )
                if DEBUG_INSTRUMENTATION_AVAILABLE:
                    try:
                        log_mapping_decision("hkjc_to_titan_attempt", {
                            "hkjc_home": hkjc.get("home_team"),
                            "hkjc_away": hkjc.get("away_team"),
                            "titan_id": titan.get("match_id"),
                            "titan_home": titan.get("home_team"),
                            "titan_away": titan.get("away_team"),
                            "avg_sim": avg_sim,
                            "is_swapped": is_swapped,
                        })
                    except Exception:
                        pass

                if avg_sim >= self.min_similarity_threshold:
                    score = avg_sim * 100
                    if score > best_score:
                        best_score = score
                        best_match = titan
                        best_is_swapped = is_swapped

            if best_match and best_score >= self.min_similarity_threshold * 100:
                titan_id = best_match['match_id']
                macau = self.macau_mapping.get(titan_id)
                source_type = classify(hkjc, best_match, macau)
                matched_item = {
                    "source_coverage": source_type,
                    "hkjc_match": {
                        "match_id": hkjc.get('match_id', ''),
                        "event_id": hkjc.get('event_id', ''),
                        "home_team": hkjc['home_team'],
                        "away_team": hkjc['away_team'],
                        "match_time": hkjc.get('match_time_original', ''),
                        "tournament": hkjc.get('tournament', '')
                    },
                    "titan_match": {
                        "match_id": titan_id,
                        "home_team": best_match['home_team'],
                        "away_team": best_match['away_team'],
                        "scheduled_time": best_match.get('scheduled_time_original', ''),
                        "league": hkjc.get('tournament', ''),
                        "status": best_match.get('match_status', '')
                    },
                    "macau_match": macau,
                    "similarity_score": best_score,
                    "teams_swapped": best_is_swapped,
                    "matched_at": datetime.now().isoformat()
                }

                # HKJC detailed odds: use bulk cache if available; else fallback to live scrape
                hkjc_event_id = hkjc.get("event_id")
                if hkjc_event_id:
                    cached_odds = self.hkjc_bulk_odds.get(hkjc_event_id)
                    if cached_odds:
                        matched_item["hkjc_detailed_odds"] = cached_odds
                        cprint(f"   📊 HKJC odds from bulk cache (Event {hkjc_event_id})", Fore.GREEN)
                    else:
                        cprint(f"   📊 Scraping HKJC All Odds live (Event {hkjc_event_id})...", Fore.CYAN)
                        detailed_odds = await hkjc_detailed_scraper.scrape(hkjc_event_id)
                        if detailed_odds:
                            matched_item["hkjc_detailed_odds"] = detailed_odds
                            cprint("   ✅ HKJC detailed odds captured", Fore.GREEN)
                        else:
                            cprint("   ⚠️ HKJC detailed odds failed", Fore.YELLOW)

                # Skip if stats already processed
                if titan_id in self.titan_stats_processed:
                    matched_item["titan_stats_available"] = False
                    matched_item["skipped_reason"] = "titan_stats_cached"
                    matched_item["ai_status"] = "skipped"
                    matched_item["ai_reason"] = "titan_stats_cached"
                    cprint("   ⏭️ Skipping Titan stats (cached from previous run)", Fore.YELLOW)
                    matched.append(matched_item)
                    self.data_quality_metrics['high_confidence_matches'] += 1
                    continue

                # Skip AI/stat scrape if already processed by AI cache
                if self.ai_cache.get(titan_id):
                    matched_item["ai_recommendation"] = {
                        "best_bet_market": "Skipped",
                        "best_bet_selection": "Already processed by DeepSeek",
                        "confidence_level": 0,
                        "brief_reasoning": "Cached AI result exists; skipping repeat call."
                    }
                    matched_item["titan_stats_available"] = False
                    matched_item["skipped_reason"] = "ai_cached"
                    matched_item["ai_status"] = "skipped"
                    matched_item["ai_reason"] = "ai_cached"
                    cprint("   ⏭️ Skipping AI/stat scrape (cached)", Fore.YELLOW)
                    matched.append(matched_item)
                    self.data_quality_metrics['high_confidence_matches'] += 1
                    continue

                if DEBUG_INSTRUMENTATION_AVAILABLE:
                    try:
                        log_mapping_decision("final_match_result", {
                            "hkjc": matched_item["hkjc_match"],
                            "titan": matched_item["titan_match"],
                            "macau": macau,
                            "source_coverage": source_type,
                            "similarity_score": best_score
                        })
                    except Exception:
                        pass

                sim_pct = best_score
                cprint(
                    f"\n✅ {source_type.upper()}: {hkjc['home_team']} vs {hkjc['away_team']} "
                    f"(similarity={sim_pct:.1f}%)",
                    Fore.GREEN
                )
                cprint(f"   HKJC: {hkjc.get('match_time_original', 'N/A')}", Fore.GREEN)
                cprint(f"   Titan: time={best_match.get('scheduled_time_original', 'N/A')}", Fore.GREEN)
                if macau:
                    cprint(f"   Macau: ID {macau.get('event_id', 'N/A')}", Fore.GREEN)

                # Titan stats + AI
                try:
                    cprint("   📊 Scraping Titan stats...", Fore.CYAN)
                    detailed_stats = await scrape_match_stats_from_analysis_page(titan_id)
                    matched_item['detailed_stats'] = detailed_stats
                    self.titan_stats_processed.add(titan_id)
                    save_cache_set(TITAN_STATS_PROCESSED_PATH, self.titan_stats_processed)

                    if detailed_stats.get('stats_available'):
                        matched_item['titan_stats_available'] = True
                        normalized = normalize_parsed_data(detailed_stats)
                        prompt = build_ai_prompt_with_availability(normalized, use_chinese=True)
                        if DEBUG_INSTRUMENTATION_AVAILABLE:
                            try:
                                save_parsed_json("ai_prompt", titan_id, {"prompt": prompt[:4000]})
                                log_info("Calling AI for titan", {"titan_id": titan_id, "available_sections": normalized.get("_meta")})
                            except Exception:
                                pass
                        cprint("   🤖 Running AI analysis...", Fore.CYAN)
                        ai_result = await perform_ai_analysis_for_match_async(normalized, call_deepseek_api_async)
                        matched_item['ai_recommendation'] = ai_result
                        self.ai_cache[titan_id] = {
                            "processed_at": datetime.now().isoformat(),
                            "hkjc_match_id": hkjc.get("match_id", ""),
                            "home_team": hkjc.get("home_team"),
                            "away_team": hkjc.get("away_team")
                        }
                        save_ai_cache(self.ai_cache)

                        if ai_result.get("ai_parsed_json"):
                            matched_item["ai_status"] = "ok"
                            matched_item["ai_reason"] = ""
                        else:
                            matched_item["ai_status"] = "no_json"
                            matched_item["ai_reason"] = ai_result.get("brief_reasoning", "No parsable JSON")

                        if ai_result and ai_result.get('best_bet_market') and ai_result.get('best_bet_selection') and ai_result.get('confidence_level', 0) > 0:
                            ai_rows_for_excel.append({
                                "match_id_titan": titan_id,
                                "match_id_hkjc": hkjc.get('match_id', ''),
                                "source_coverage": source_type,
                                "home_team": hkjc['home_team'],
                                "away_team": hkjc['away_team'],
                                "match_time": hkjc.get('match_time_original', ''),
                                "league": hkjc.get('tournament', ''),
                                "best_bet_market": ai_result.get('best_bet_market'),
                                "best_bet_selection": ai_result.get('best_bet_selection'),
                                "confidence_level": ai_result.get('confidence_level'),
                                "brief_reasoning": ai_result.get('brief_reasoning'),
                                "similarity_score": best_score,
                                "ai_analysis_timestamp": datetime.now().isoformat()
                            })
                            cprint(f"   💡 AI Recommendation: {ai_result.get('best_bet_selection')} ({ai_result.get('confidence_level')}/10)", Fore.CYAN)
                        else:
                            cprint("   ⚠️ AI returned no strong recommendation", Fore.YELLOW)
                    else:
                        matched_item['titan_stats_available'] = False
                        matched_item['skipped_reason'] = detailed_stats.get('error', 'no_stats')
                        matched_item["ai_status"] = "skipped"
                        matched_item["ai_reason"] = matched_item['skipped_reason']
                        cprint("   ⚠️ No stats available for analysis", Fore.YELLOW)
                except Exception as e:
                    logger.exception("Failed to analyze Titan match %s: %s", titan_id, e)
                    matched_item['ai_recommendation'] = {
                        "best_bet_market": "Error",
                        "best_bet_selection": "Processing error",
                        "confidence_level": 0,
                        "brief_reasoning": str(e)
                    }
                    matched_item["ai_status"] = "error"
                    matched_item["ai_reason"] = str(e)

                matched.append(matched_item)
                self.data_quality_metrics['high_confidence_matches'] += 1
            else:
                unmatched_hkjc.append(hkjc)
                if len(unmatched_hkjc) <= 3:
                    reason = "No similar teams found"
                    if best_match:
                        reason = f"Similarity {best_score:.1f}% below threshold {self.min_similarity_threshold*100}%"
                    cprint(f"\n❌ NO MATCH: {hkjc['home_team']} vs {hkjc['away_team']}", Fore.RED)
                    cprint(f"   Reason: {reason}", Fore.RED)

        self.matched_games = matched
        self.unmatched_games = unmatched_hkjc
        save_ai_cache(self.ai_cache)
        save_cache_set(HKJC_ODDS_PROCESSED_PATH, self.hkjc_odds_processed)
        save_cache_set(TITAN_STATS_PROCESSED_PATH, self.titan_stats_processed)

        cprint(f"\n📊 FINAL RESULTS:", Fore.CYAN)
        cprint(f"   ✅ Matched games: {len(matched)}", Fore.GREEN)
        cprint(f"   ❌ Unmatched HKJC games: {len(unmatched_hkjc)}", Fore.RED)
        if hkjc_matches:
            success_rate = len(matched) / len(hkjc_matches) * 100
            cprint(f"   📈 Success rate: {success_rate:.1f}%", Fore.CYAN)

        if ai_rows_for_excel:
            excel_filename = f"ai_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            try:
                save_recommendations_to_excel(ai_rows_for_excel, excel_filename)
                cprint(f"\n📊 AI recommendations saved to: {excel_filename}", Fore.CYAN)
            except Exception as e:
                cprint(f"❌ Failed saving AI Excel: {e}", Fore.RED)

        # Export all sources in one sheet (matched first)
        self.save_all_sources_ordered_excel(hkjc_matches, titan_matches, macau_odds, matched, unmatched_hkjc)

        return matched, unmatched_hkjc

    # Reporting helpers unchanged
    def generate_detailed_report(self) -> Dict:
        report = {
            "summary": {
                "total_matched": len(self.matched_games),
                "total_unmatched": len(self.unmatched_games),
                "data_quality_metrics": self.data_quality_metrics
            },
            "top_matches": sorted(self.matched_games, key=lambda x: x.get('similarity_score', 0), reverse=True)[:5],
            "common_issues": [],
            "recommendations": []
        }
        if self.data_quality_metrics['total_hkjc_matches'] == 0:
            report['common_issues'].append("No matches found on HKJC")
        if self.data_quality_metrics['total_titan_matches'] == 0:
            report['common_issues'].append("No matches found on Titan007")
        return report

    def save_report(self, report: Dict, filename: Optional[str] = None):
        if not filename:
            filename = f"detailed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        cprint(f"📊 Detailed report saved to: {filename}", Fore.CYAN)

    # 1:1 comparison export to Excel (no Cartesian explosion)
    def save_comparison_excel(self, filename: Optional[str] = None):
        if not filename:
            filename = f"all_scraped_matches_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        if not self.matched_games and not self.unmatched_games:
            cprint("⚠️ No data available for comparison export", Fore.YELLOW)
            return

        rows = []
        for m in self.matched_games:
            h = m.get("hkjc_match", {})
            t = m.get("titan_match", {})
            mc = m.get("macau_match", {}) or {}
            rows.append({
                "hkjc_match_id": h.get("match_id"),
                "hkjc_event_id": h.get("event_id"),
                "hkjc_home": h.get("home_team"),
                "hkjc_away": h.get("away_team"),
                "hkjc_time": h.get("match_time"),
                "titan_match_id": t.get("match_id"),
                "titan_home": t.get("home_team"),
                "titan_away": t.get("away_team"),
                "titan_time": t.get("scheduled_time"),
                "macau_event_id": mc.get("event_id"),
                "macau_time": mc.get("time"),
                "macau_competition": mc.get("competition"),
                "macau_home": mc.get("home_team"),
                "macau_away": mc.get("away_team"),
                "similarity_score": m.get("similarity_score"),
                "source_coverage": m.get("source_coverage"),
            })
        for h in self.unmatched_games:
            rows.append({
                "hkjc_match_id": h.get("match_id"),
                "hkjc_event_id": h.get("event_id"),
                "hkjc_home": h.get("home_team"),
                "hkjc_away": h.get("away_team"),
                "hkjc_time": h.get("match_time_original"),
                "titan_match_id": None,
                "titan_home": None,
                "titan_away": None,
                "titan_time": None,
                "macau_event_id": None,
                "macau_time": None,
                "macau_competition": None,
                "macau_home": None,
                "macau_away": None,
                "similarity_score": None,
                "source_coverage": "hkjc_only",
            })
        df = pd.DataFrame(rows)
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Comparison", index=False)
        cprint(f"📋 Comparison saved to: {filename}", Fore.CYAN)

    # AI results to Excel
    def save_ai_results_excel(self, matched_games: List[Dict], filename: Optional[str] = None):
        if not filename:
            filename = f"matched_games_with_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        rows = []
        for m in matched_games:
            h = m.get("hkjc_match", {})
            t = m.get("titan_match", {})
            ai = m.get("ai_recommendation", {}) or {}
            rows.append({
                "hkjc_match_id": h.get("match_id"),
                "hkjc_event_id": h.get("event_id"),
                "hkjc_home": h.get("home_team"),
                "hkjc_away": h.get("away_team"),
                "titan_match_id": t.get("match_id"),
                "titan_home": t.get("home_team"),
                "titan_away": t.get("away_team"),
                "best_bet_market": ai.get("best_bet_market"),
                "best_bet_selection": ai.get("best_bet_selection"),
                "confidence_level": ai.get("confidence_level"),
                "brief_reasoning": ai.get("brief_reasoning"),
                "similarity_score": m.get("similarity_score"),
                "source_coverage": m.get("source_coverage"),
                "ai_status": m.get("ai_status"),
                "ai_reason": m.get("ai_reason"),
            })
        df = pd.DataFrame(rows)
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="AI_Results", index=False)
        cprint(f"💾 AI results saved to: {filename}", Fore.CYAN)

    def save_comparison_csv(self, filename: Optional[str] = None):
        # Deprecated: kept for compatibility, but uses Excel exporter now
        self.save_comparison_excel(filename.replace(".csv", ".xlsx") if filename else None)

    # All sources in one sheet, matched groups first, then unmatched
    def save_all_sources_ordered_excel(self,
                                       hkjc_matches: List[Dict],
                                       titan_matches: List[Dict],
                                       macau_matches: List[Dict],
                                       matched_games: List[Dict],
                                       unmatched_hkjc: List[Dict],
                                       filename: Optional[str] = None):
        if not filename:
            filename = f"all_sources_ordered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        rows = []
        matched_titan_ids = set()
        matched_macau_ids = set()
        matched_hkjc_ids = set()

        # Matched groups first
        for idx, m in enumerate(matched_games, 1):
            group_id = f"M{idx}"
            h = m.get("hkjc_match", {}) or {}
            t = m.get("titan_match", {}) or {}
            mc = m.get("macau_match", {}) or {}

            matched_hkjc_ids.add(h.get("match_id"))
            matched_titan_ids.add(t.get("match_id"))
            if mc.get("event_id"):
                matched_macau_ids.add(mc.get("event_id"))

            rows.append({
                "group": group_id, "source": "HKJC", "matched_flag": True,
                "match_id": h.get("match_id"), "event_id": h.get("event_id"),
                "home": h.get("home_team"), "away": h.get("away_team"),
                "time": h.get("match_time"), "competition": h.get("tournament"),
                "similarity_score": m.get("similarity_score"), "coverage": m.get("source_coverage"),
            })
            rows.append({
                "group": group_id, "source": "Titan007", "matched_flag": True,
                "match_id": t.get("match_id"), "event_id": None,
                "home": t.get("home_team"), "away": t.get("away_team"),
                "time": t.get("scheduled_time"), "competition": t.get("league"),
                "similarity_score": m.get("similarity_score"), "coverage": m.get("source_coverage"),
            })
            if mc:
                rows.append({
                    "group": group_id, "source": "MacauSlot", "matched_flag": True,
                    "match_id": mc.get("event_id"), "event_id": mc.get("event_id"),
                    "home": mc.get("home_team"), "away": mc.get("away_team"),
                    "time": mc.get("time"), "competition": mc.get("competition"),
                    "similarity_score": m.get("similarity_score"), "coverage": m.get("source_coverage"),
                })

        # Unmatched HKJC
        for h in unmatched_hkjc:
            rows.append({
                "group": "U_HKJC", "source": "HKJC", "matched_flag": False,
                "match_id": h.get("match_id"), "event_id": h.get("event_id"),
                "home": h.get("home_team"), "away": h.get("away_team"),
                "time": h.get("match_time_original"), "competition": h.get("tournament", ""),
                "similarity_score": None, "coverage": "hkjc_only",
            })

        # Unmatched Titan
        for t in titan_matches:
            if t.get("match_id") in matched_titan_ids:
                continue
            rows.append({
                "group": "U_Titan", "source": "Titan007", "matched_flag": False,
                "match_id": t.get("match_id"), "event_id": None,
                "home": t.get("home_team"), "away": t.get("away_team"),
                "time": t.get("scheduled_time_original"), "competition": t.get("league"),
                "similarity_score": None, "coverage": "titan_only",
            })

        # Unmatched Macau
        for mc in macau_matches:
            if mc.get("event_id") in matched_macau_ids:
                continue
            rows.append({
                "group": "U_Macau", "source": "MacauSlot", "matched_flag": False,
                "match_id": mc.get("event_id"), "event_id": mc.get("event_id"),
                "home": mc.get("home_team"), "away": mc.get("away_team"),
                "time": mc.get("time"), "competition": mc.get("competition"),
                "similarity_score": None, "coverage": "macau_only",
            })

        df = pd.DataFrame(rows)
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="AllSources", index=False)
        cprint(f"📋 All sources (matched first) saved to: {filename}", Fore.CYAN)

# ---------------------------------------------------------------------------
# Excel saver (AI recommendations grouped)
# ---------------------------------------------------------------------------
def save_recommendations_to_excel(recommendations: List[Dict], filename: str):
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df_all = pd.DataFrame(recommendations)
        df_all.to_excel(writer, sheet_name='Summary', index=False)
        if not df_all.empty:
            for market in df_all['best_bet_market'].unique():
                df_m = df_all[df_all['best_bet_market'] == market]
                safe_sheet = re.sub(r'[\\/*?:\[\]]', '_', str(market))[:31]
                try:
                    df_m.to_excel(writer, sheet_name=safe_sheet, index=False)
                except Exception:
                    pass
    logger.info("💾 AI recommendations saved to Excel file: %s", filename)

# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
async def main():
    cprint("🚀 LIVE MATCH CROSS-REFERENCER — WITH HKJC ALL ODDS + MACAU SLOT", Fore.WHITE)
    cprint("=" * 80, Fore.WHITE)

    matcher = LiveMatchMatcher(min_similarity_threshold=0.75, time_tolerance_minutes=30, prioritize_similarity=True)
    matched_games, unmatched = await matcher.find_matching_games()

    report = matcher.generate_detailed_report()
    matcher.save_report(report)
    matcher.save_comparison_excel()

    if matched_games:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_json = f"matched_games_with_ai_analysis_{ts}.json"
        with open(filename_json, "w", encoding="utf-8") as f:
            json.dump(matched_games, f, ensure_ascii=False, indent=2)
        cprint(f"\n💾 Saved {len(matched_games)} matched games (with AI analysis) to: {filename_json}", Fore.CYAN)
        matcher.save_ai_results_excel(matched_games)

    if unmatched:
        analysis_file = f"unmatched_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump({
                "unmatched_count": len(unmatched),
                "sample_unmatched": unmatched[:10],
                "analysis_time": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        cprint(f"💾 Saved unmatched analysis to: {analysis_file}", Fore.CYAN)

    cprint("\n✅ Process complete!", Fore.GREEN)
    cprint(f"📊 Summary: {len(matched_games)} matches analyzed with AI recommendations", Fore.CYAN)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        cprint("Interrupted by user.", Fore.YELLOW)
    except Exception as e:
        logger.exception("Fatal error in main: %s", e)
