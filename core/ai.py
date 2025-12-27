#!/usr/bin/env python3
"""
AI integration module for match analysis using DeepSeek API.
Handles prompt building, API calls, and response parsing.
"""

import asyncio
import json
import logging
import os
import re
from typing import Dict, Any, Optional, Tuple

import httpx

from .utils import EXPECTED_TOP_LEVEL_SECTIONS

logger = logging.getLogger("matcher")

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
DEEPSEEK_TIMEOUT = float(os.getenv("DEEPSEEK_TIMEOUT", "30"))
DEEPSEEK_RETRIES = int(os.getenv("DEEPSEEK_RETRIES", "3"))

if not DEEPSEEK_API_KEY:
    logger.warning("DEEPSEEK_API_KEY not set. AI functionality will fail unless you set the env var.")


async def call_deepseek_api(prompt: str, timeout: int = None, max_retries: int = None) -> str:
    """
    Call DeepSeek API with retry logic.
    
    Args:
        prompt: The prompt to send to the API
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        
    Returns:
        API response text
    """
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


def normalize_parsed_data(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize parsed data structure for AI analysis.
    
    Args:
        parsed: Raw parsed data dictionary
        
    Returns:
        Normalized data with metadata
    """
    merged = dict(parsed)
    if isinstance(parsed.get("sections"), dict):
        for k, v in parsed["sections"].items():
            if k not in merged or merged.get(k) is None:
                merged[k] = v

    normalized: Dict[str, Any] = {}
    missing: list = []
    available: list = []

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
    """Check if the normalized stats contain meaningful data for AI analysis."""
    meta = normalized_stats.get("_meta", {})
    available = meta.get("available_sections", [])
    if available:
        return True
    rr = normalized_stats.get("recent10_ratings_parsed") or {}
    if rr.get("home_recent_ratings") or rr.get("away_recent_ratings") or rr.get("home_recent_average") or rr.get("away_recent_average"):
        return True
    return False


def build_ai_prompt_with_availability(normalized_data: Dict[str, Any], use_chinese: bool = True) -> str:
    """
    Build AI prompt based on available data sections.
    
    Args:
        normalized_data: Normalized match data
        use_chinese: Whether to use Chinese in the prompt
        
    Returns:
        Formatted prompt string
    """
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
    """
    Parse AI response to extract JSON.
    
    Args:
        text: Raw AI response text
        
    Returns:
        Tuple of (parsed_dict, raw_json_string)
    """
    if not text:
        return None, None
    try:
        s = text.strip()
        if s.startswith("{") and s.endswith("}"):
            return json.loads(s), s
        m = re.search(r'(\{(?:.|\s)*\})', text)
        if m:
            candidate = m.group(1)
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


async def perform_ai_analysis_for_match(
    normalized_stats: Dict[str, Any],
    call_deepseek_api_fn,
    max_retries: int = 2,
    short_circuit_when_no_data: bool = True
) -> Dict[str, Any]:
    """
    Perform AI analysis for a match.
    
    Args:
        normalized_stats: Normalized match statistics
        call_deepseek_api_fn: Function to call DeepSeek API
        max_retries: Maximum retry attempts
        short_circuit_when_no_data: Skip AI if no meaningful data
        
    Returns:
        Dictionary with AI analysis results
    """
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

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            ai_text = await call_deepseek_api_fn(prompt)
            result["ai_raw_response"] = ai_text
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
