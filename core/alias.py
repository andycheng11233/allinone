#!/usr/bin/env python3
"""
Alias management system for team and league names.
Handles loading, saving, and resolving aliases across different sources.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("matcher")

# File paths
ALIAS_JSON_PATH = Path("alias.json")
UNALIAS_PENDING_PATH = Path("unalias_pending.json")

# In-memory alias structures (rebuilt from alias.json)
alias_data: Dict[str, Dict[str, Dict[str, Any]]] = {"teams": {}, "leagues": {}}
alias_modified = False
ALIAS_TABLE: Dict[str, set] = {}          # team alias lookup
LEAGUE_ALIAS_TABLE: Dict[str, set] = {}   # league alias lookup

ALIAS_SOURCE_PRIORITY = ["hkjc", "macauslot", "titan"]


def load_alias_table_from_json() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load alias table from JSON file."""
    if not ALIAS_JSON_PATH.exists():
        return {"teams": {}, "leagues": {}}
    try:
        data = json.loads(ALIAS_JSON_PATH.read_text(encoding="utf-8"))
        # Normalize structure
        for cat in ("teams", "leagues"):
            for canon, entry in list(data.get(cat, {}).items()):
                entry.setdefault("variants", [])
                entry.setdefault("sources", {})
        return {"teams": data.get("teams", {}), "leagues": data.get("leagues", {})}
    except Exception as e:
        logger.warning("Failed to load alias.json: %s", e)
        return {"teams": {}, "leagues": {}}


def save_alias_table_if_needed():
    """Save alias table to JSON file if modified."""
    global alias_modified
    if not alias_modified:
        return
    try:
        ALIAS_JSON_PATH.write_text(json.dumps(alias_data, ensure_ascii=False, indent=2), encoding="utf-8")
        alias_modified = False
    except Exception as e:
        logger.warning("Failed to save alias.json: %s", e)


def rebuild_alias_lookup():
    """Rebuild the in-memory alias lookup tables."""
    global ALIAS_TABLE, LEAGUE_ALIAS_TABLE
    ALIAS_TABLE = {}
    LEAGUE_ALIAS_TABLE = {}
    for canon, entry in alias_data.get("teams", {}).items():
        variants = set()
        for v in entry.get("variants", []):
            if v:
                variants.add(v.strip().lower())
        variants.add(canon.strip().lower())
        ALIAS_TABLE[canon.strip().lower()] = variants
    for canon, entry in alias_data.get("leagues", {}).items():
        variants = set()
        for v in entry.get("variants", []):
            if v:
                variants.add(v.strip().lower())
        variants.add(canon.strip().lower())
        LEAGUE_ALIAS_TABLE[canon.strip().lower()] = variants


def resolve_alias(name: str) -> str:
    """Resolve a team name to its canonical form."""
    key = name.strip().lower()
    for canon, variants in ALIAS_TABLE.items():
        if key == canon or key in variants:
            return canon
    return key


def resolve_league_alias(name: str) -> str:
    """Resolve a league name to its canonical form."""
    key = name.strip().lower()
    for canon, variants in LEAGUE_ALIAS_TABLE.items():
        if key == canon or key in variants:
            return canon
    return key


def upsert_alias(kind: str, raw_name: str, source: str):
    """
    Insert or update an alias entry.
    
    Args:
        kind: "teams" or "leagues"
        source: "hkjc" | "macauslot" | "titan"
        
    Canonical preference: hkjc > macauslot > titan. If HKJC shows up later, canonical is promoted.
    """
    from .utils import strip_accents
    import re
    
    global alias_modified

    if not raw_name:
        return
    if kind not in ("teams", "leagues"):
        return

    # Import normalization functions locally to avoid circular imports
    if kind == "teams":
        base_norm = normalize_team_name(raw_name, apply_alias=False)
    else:
        base_norm = normalize_league(raw_name, apply_alias=False)

    if not base_norm:
        return

    data = alias_data.setdefault(kind, {})
    raw_lower = raw_name.strip().lower()

    # Find existing canonical containing raw_name
    found_canon = None
    for canon, entry in data.items():
        variants = [v.strip().lower() for v in entry.get("variants", [])]
        if raw_lower == canon or raw_lower in variants:
            found_canon = canon
            break
    if not found_canon and base_norm in data:
        found_canon = base_norm

    # Create or promote canonical
    if not found_canon:
        canon = base_norm
        data.setdefault(canon, {"variants": [], "sources": {}})
        entry = data[canon]
    else:
        canon = found_canon
        entry = data[canon]
        # Promote to HKJC if higher priority and new base_norm not present
        if source == "hkjc" and canon != base_norm and base_norm not in data:
            data[base_norm] = entry
            del data[canon]
            canon = base_norm
            entry = data[canon]

    # Add variants
    variants = entry.get("variants", [])
    def add_var(val: str):
        if val and val not in variants:
            variants.append(val)
    add_var(raw_name)
    add_var(base_norm)
    entry["variants"] = variants

    # Add source variants
    srcs = entry.get("sources", {})
    lst = srcs.get(source, [])
    if raw_name not in lst:
        lst.append(raw_name)
    if base_norm not in lst:
        lst.append(base_norm)
    srcs[source] = lst
    entry["sources"] = srcs
    data[canon] = entry
    alias_data[kind] = data
    alias_modified = True
    rebuild_alias_lookup()


def normalize_team_name(name: str, apply_alias: bool = True) -> str:
    """Normalize a team name for comparison."""
    from .utils import strip_accents
    import re
    
    if not name:
        return ""
    name = strip_accents(name)
    name = name.lower()
    name = re.sub(r'\[\d+\]', '', name)
    name = re.sub(r'\(中\)', '', name)
    name = re.sub(r'(女足|女子)$', '', name)
    name = re.sub(r'\b(fc|cf|sc|afc|cfc)\b', '', name)
    name = re.sub(r'[^a-z0-9\u4e00-\u9fff]+', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    if apply_alias:
        name = resolve_alias(name)
    return name


def normalize_league(text: str, apply_alias: bool = True) -> str:
    """Normalize a league name for comparison."""
    from .utils import strip_accents
    import re
    
    if not text:
        return ""
    t = strip_accents(text).lower()
    t = re.sub(r'[^a-z0-9\u4e00-\u9fff]+', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    if apply_alias:
        t = resolve_league_alias(t)
    return t


def append_unalias_pending(kind: str, seen: str, source: str = "", context: str = "", suggested_canon: str = ""):
    """Append an unresolved name to the pending list for manual review."""
    if not seen:
        return
    UNALIAS_PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        pending = json.loads(UNALIAS_PENDING_PATH.read_text(encoding="utf-8")) if UNALIAS_PENDING_PATH.exists() else {"teams": [], "leagues": []}
    except Exception:
        pending = {"teams": [], "leagues": []}
    bucket = pending.get(kind, [])
    if not any((entry.get("seen") == seen and entry.get("source") == source) for entry in bucket):
        bucket.append({"seen": seen, "source": source, "context": context, "suggested_canon": suggested_canon})
    pending[kind] = bucket
    UNALIAS_PENDING_PATH.write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")


# Load aliases at start
alias_data = load_alias_table_from_json()
rebuild_alias_lookup()
