#!/usr/bin/env python3
"""
Match matcher module.

This module contains the main LiveMatchMatcher class that:
- Coordinates scraping from multiple sources
- Matches games across HKJC, Titan007, and MacauSlot
- Applies AI analysis to matched games
- Generates reports and exports results

TODO: Complete extraction of LiveMatchMatcher from script A.
This is a demonstration stub showing the intended structure.
"""

import asyncio
import json
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

from .utils import cprint, logger, Fore, AI_CACHE_PATH, HKJC_ODDS_PROCESSED_PATH, TITAN_STATS_PROCESSED_PATH
from .alias import normalize_team_name, normalize_league, upsert_alias, save_alias_table_if_needed, append_unalias_pending
from .ai import call_deepseek_api, normalize_parsed_data, has_meaningful_data_for_ai, perform_ai_analysis_for_match
from scrapers.hkjc import HKJCHomeScraper, HKJCDetailedOddsScraper, HKJCBulkOddsCollector, load_hkjc_odds_from_disk
from scrapers.titan import TitanStatsScraper, load_titan_stats_from_disk
from scrapers.macauslot import MacauSlotOddsScraper


def name_similarity(a: str, b: str) -> float:
    """Calculate similarity between two team names."""
    na = normalize_team_name(a)
    nb = normalize_team_name(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def league_bonus(l1: str, l2: str, bonus: float = 0.05) -> float:
    """Bonus score for matching leagues."""
    if not l1 or not l2:
        return 0.0
    return bonus if normalize_league(l1) == normalize_league(l2) else 0.0


def token_overlap_score(a: str, b: str) -> float:
    """Calculate token overlap score between two strings."""
    ta = set(normalize_team_name(a).split())
    tb = set(normalize_team_name(b).split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / max(len(ta), len(tb))


class LiveMatchMatcher:
    """
    Main class for matching games across multiple betting platforms.
    
    Coordinates:
    1. Data collection from HKJC, Titan007, MacauSlot
    2. Team name matching with fuzzy logic
    3. Time-based filtering
    4. AI analysis integration
    5. Report generation
    """
    
    def __init__(self, 
                 min_similarity_threshold: float = 0.70,
                 time_tolerance_minutes: int = 30,
                 prioritize_similarity: bool = True,
                 hk_titan_time_tolerance: int = 45,
                 titan_macau_time_tolerance: int = 10):
        """
        Initialize the matcher.
        
        Args:
            min_similarity_threshold: Minimum name similarity to consider a match
            time_tolerance_minutes: Time window for matching (minutes)
            prioritize_similarity: Prioritize name similarity over time matching
            hk_titan_time_tolerance: HKJC-Titan time tolerance (minutes)
            titan_macau_time_tolerance: Titan-Macau time tolerance (minutes)
        """
        self.min_similarity_threshold = min_similarity_threshold
        self.time_tolerance_minutes = time_tolerance_minutes
        self.prioritize_similarity = prioritize_similarity
        self.hk_titan_time_tolerance = hk_titan_time_tolerance
        self.titan_macau_time_tolerance = titan_macau_time_tolerance
        
        # Initialize scrapers
        self.titan_scraper = TitanStatsScraper()
        
        # Data storage
        self.matched_games: List[Dict] = []
        self.unmatched_games: List[Dict] = []
        self.raw_hkjc_matches: List[Dict] = []
        self.raw_titan_matches: List[Dict] = []
        self.macau_mapping: Dict[str, Dict] = {}
        
        # Caches
        self.ai_cache: Dict = self.load_ai_cache()
        self.hkjc_bulk_odds: Dict = load_hkjc_odds_from_disk()
        self.hkjc_odds_processed: set = self.load_cache_set(HKJC_ODDS_PROCESSED_PATH)
        self.titan_stats_processed: set = self.load_cache_set(TITAN_STATS_PROCESSED_PATH)
        
        # Metrics
        self.data_quality_metrics = {
            'total_hkjc_matches': 0,
            'total_titan_matches': 0,
            'high_confidence_matches': 0,
            'potential_matches_checked': 0
        }
    
    @staticmethod
    def load_ai_cache() -> Dict[str, Any]:
        """Load AI analysis cache from disk."""
        try:
            if AI_CACHE_PATH.exists():
                return json.loads(AI_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to load AI cache: %s", e)
        return {}
    
    @staticmethod
    def save_ai_cache(cache: Dict[str, Any]):
        """Save AI analysis cache to disk."""
        try:
            AI_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            AI_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save AI cache: %s", e)
    
    @staticmethod
    def load_cache_set(path: Path, as_str: bool = False) -> set:
        """Load a set from a JSON file."""
        try:
            if path.exists():
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if as_str:
                    return set(str(x) for x in loaded)
                return set(loaded)
        except Exception:
            pass
        return set()
    
    @staticmethod
    def save_cache_set(path: Path, data: set):
        """Save a set to a JSON file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(sorted(list(data))), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save cache set to %s: %s", path, e)
    
    async def find_matching_games(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Main method to find and match games across all platforms.
        
        Returns:
            Tuple of (matched_games, unmatched_games)
        """
        # TODO: Complete implementation
        # Reference: Lines 2389+ in script A (find_matching_games method)
        
        cprint("\n" + "=" * 80, Fore.WHITE)
        cprint("🔍 FINDING MATCHES (HKJC + Titan007 + Macau Slot)", Fore.WHITE)
        cprint("=" * 80, Fore.WHITE)
        
        cprint("\n   [STUB] Would execute full matching workflow", Fore.YELLOW)
        cprint("   Steps: HKJC bulk odds → Macau scraping → HKJC list → Titan list", Fore.YELLOW)
        cprint("   → Alias ingestion → Matching → AI analysis → Report generation", Fore.YELLOW)
        
        return [], []
    
    def generate_detailed_report(self) -> Dict:
        """Generate a detailed report of matching results."""
        report = {
            "summary": {
                "total_matched": len(self.matched_games),
                "total_unmatched": len(self.unmatched_games),
                "data_quality_metrics": self.data_quality_metrics
            },
        }
        return report
    
    def save_report(self, report: Dict, filename: Optional[str] = None):
        """Save the report to a JSON file."""
        if not filename:
            filename = f"detailed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        cprint(f"📊 Detailed report saved to: {filename}", Fore.CYAN)
    
    def save_ai_results_excel(self, matched_games: List[Dict], filename: Optional[str] = None):
        """Save AI analysis results to Excel."""
        if not filename:
            filename = f"matched_games_with_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        # TODO: Complete implementation
        # Reference: Lines 3032+ in script A (save_ai_results_excel method)
        
        cprint(f"   [STUB] Would save AI results to: {filename}", Fore.YELLOW)


# Example usage
if __name__ == "__main__":
    async def test():
        """Test the matcher."""
        cprint("Testing LiveMatchMatcher (stub implementation)", Fore.CYAN)
        
        matcher = LiveMatchMatcher(
            min_similarity_threshold=0.70,
            time_tolerance_minutes=30
        )
        
        matched, unmatched = await matcher.find_matching_games()
        
        report = matcher.generate_detailed_report()
        matcher.save_report(report)
        
        cprint("Test complete. Implement TODO items to enable full functionality.", Fore.GREEN)
    
    asyncio.run(test())
