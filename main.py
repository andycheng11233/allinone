#!/usr/bin/env python3
"""
Main entry point for the match analysis system.

This orchestrates the entire workflow:
1. Scrape data from HKJC, Titan007, and MacauSlot
2. Match games across platforms using fuzzy matching
3. Run AI analysis on matched games
4. Generate comprehensive reports

Usage:
    python main.py

Environment Variables:
    DEEPSEEK_API_KEY: API key for DeepSeek AI (required for AI analysis)
    DEEPSEEK_API_URL: Custom API endpoint (optional)
    DEEPSEEK_TIMEOUT: Request timeout in seconds (default: 30)
    DEEPSEEK_RETRIES: Number of retry attempts (default: 3)
"""

import asyncio
import json
import sys
from datetime import datetime

from core.utils import cprint, logger, Fore, Style
from core.matcher import LiveMatchMatcher


async def main():
    """Main execution function."""
    cprint("=" * 80, Fore.WHITE, Style.BRIGHT)
    cprint("🚀 LIVE MATCH CROSS-REFERENCER", Fore.WHITE, Style.BRIGHT)
    cprint("    HKJC + Titan007 + MacauSlot + AI Analysis", Fore.WHITE)
    cprint("=" * 80, Fore.WHITE, Style.BRIGHT)
    cprint("", Fore.WHITE)
    
    try:
        # Initialize the matcher with configuration
        matcher = LiveMatchMatcher(
            min_similarity_threshold=0.70,
            time_tolerance_minutes=30,
            prioritize_similarity=True,
            hk_titan_time_tolerance=45,
            titan_macau_time_tolerance=10
        )
        
        cprint("📋 Configuration:", Fore.CYAN)
        cprint(f"   Min similarity threshold: {matcher.min_similarity_threshold}", Fore.CYAN)
        cprint(f"   Time tolerance: {matcher.time_tolerance_minutes} minutes", Fore.CYAN)
        cprint(f"   HKJC-Titan tolerance: {matcher.hk_titan_time_tolerance} minutes", Fore.CYAN)
        cprint(f"   Titan-Macau tolerance: {matcher.titan_macau_time_tolerance} minutes", Fore.CYAN)
        cprint("", Fore.WHITE)
        
        # Execute the matching workflow
        cprint("🔄 Starting match analysis workflow...", Fore.BLUE, Style.BRIGHT)
        matched_games, unmatched = await matcher.find_matching_games()
        
        # Generate and save reports
        cprint("\n📊 Generating reports...", Fore.BLUE)
        report = matcher.generate_detailed_report()
        matcher.save_report(report)
        
        # Save matched games with AI analysis
        if matched_games:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save JSON
            filename_json = f"matched_games_with_ai_analysis_{ts}.json"
            with open(filename_json, "w", encoding="utf-8") as f:
                json.dump(matched_games, f, ensure_ascii=False, indent=2)
            cprint(f"\n💾 Saved {len(matched_games)} matched games to: {filename_json}", Fore.GREEN)
            
            # Save Excel
            matcher.save_ai_results_excel(matched_games)
        else:
            cprint("\n⚠️ No matched games to save", Fore.YELLOW)
        
        # Save unmatched analysis
        if unmatched:
            analysis_file = f"unmatched_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump({
                    "unmatched_count": len(unmatched),
                    "sample_unmatched": unmatched[:10],
                    "analysis_time": datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
            cprint(f"💾 Saved unmatched analysis to: {analysis_file}", Fore.CYAN)
        
        # Summary
        cprint("\n" + "=" * 80, Fore.WHITE)
        cprint("✅ ANALYSIS COMPLETE!", Fore.GREEN, Style.BRIGHT)
        cprint("=" * 80, Fore.WHITE)
        cprint(f"📊 Summary:", Fore.CYAN)
        cprint(f"   Matched games: {len(matched_games)}", Fore.GREEN)
        cprint(f"   Unmatched games: {len(unmatched)}", Fore.YELLOW if unmatched else Fore.GREEN)
        cprint(f"   AI recommendations: {len([m for m in matched_games if m.get('ai_recommendation')])}", Fore.CYAN)
        cprint("", Fore.WHITE)
        
        return 0
        
    except KeyboardInterrupt:
        cprint("\n⚠️ Process interrupted by user", Fore.YELLOW)
        return 1
        
    except Exception as e:
        logger.exception("Fatal error in main: %s", e)
        cprint(f"\n❌ Fatal error: {e}", Fore.RED, Style.BRIGHT)
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        logger.exception("Unhandled exception: %s", e)
        cprint(f"\n❌ Unhandled exception: {e}", Fore.RED, Style.BRIGHT)
        sys.exit(1)
