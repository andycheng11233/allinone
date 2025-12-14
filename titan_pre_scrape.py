#!/usr/bin/env python3
"""
Titan007 Pre-Scrape Script

This script bulk-scrapes Titan007 match statistics and saves them to disk
for later reuse by the allineone matcher. This reduces load times and avoids
redundant scraping.

CLI Usage:
    python3 titan_pre_scrape.py [--max-matches N] [--output-dir PATH]

Python API:
    from titan_pre_scrape import scrape_all, load_titan_stats_for_today
    
    # Scrape all matches and save to disk
    await scrape_all(max_matches=50, output_dir="titan/stats")
    
    # Load pre-scraped stats for today
    stats = load_titan_stats_for_today(stats_dir="titan/stats")
"""

import asyncio
import json
import logging
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup, Tag

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("titan_pre_scrape")


# ==================== Helper Functions (reused from allineone) ====================

def extract_ratings_or_average_from_text(page_text: str) -> Tuple[Optional[float], List[float]]:
    """Extract player ratings from Titan page text."""
    if not page_text:
        return None, []
    
    # Look for explicit average rating
    m = re.search(r'平均評分[:：]?\s*([0-9]{1,2}\.[0-9]{1,2})', page_text)
    if m:
        try:
            val = float(m.group(1))
            if 0.0 <= val <= 10.0:
                return val, [val]
        except Exception:
            pass
    
    # Look for recent 10 matches ratings
    m2 = re.search(r'(?:主隊|客隊)?近10場平均評分[:：]?\s*([0-9\.\s]{5,200})', page_text)
    if m2:
        snippet = m2.group(1)
        parsed = parse_decimal_tokens_from_concatenated(snippet)
        if parsed:
            avg = sum(parsed) / len(parsed)
            return avg, parsed
    
    # Fallback: extract all decimal ratings
    all_decimals = parse_decimal_tokens_from_concatenated(page_text)
    if all_decimals:
        chosen = all_decimals[:10]
        avg = sum(chosen) / len(chosen)
        return avg, chosen
    
    return None, []


def parse_decimal_tokens_from_concatenated(text: str) -> List[float]:
    """Parse decimal ratings from text."""
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


def extract_table_data_from_real_table(table_elem: Tag) -> List[Dict[str, str]]:
    """Extract structured data from a table element."""
    rows = table_elem.find_all('tr')
    if not rows:
        return []
    
    # Find header row
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
    
    # Extract headers
    headers = []
    for cell in header_row.find_all(['th', 'td']):
        text = cell.get_text(strip=True)
        clean_text = re.sub(r'\s+', ' ', text).strip() if text else f"col_{len(headers)}"
        headers.append(clean_text)
    
    if len(headers) < 2 or len(headers) > 30:
        return []
    
    # Extract data rows
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
    """Extract data from a table, handling nested tables."""
    rows = table_elem.find_all('tr')
    if not rows:
        return []
    
    first_row_cells = rows[0].find_all(['th', 'td'])
    
    # If too many columns, look for nested tables
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


# ==================== Titan Match List Scraper ====================

async def scrape_titan_match_list() -> List[Dict[str, Any]]:
    """
    Scrape the list of today's matches from Titan007 index page.
    
    Returns:
        List of match dictionaries with match_id, home_team, away_team, etc.
    """
    matches = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            logger.info("🌐 Loading Titan007 match list...")
            await page.goto("https://live.titan007.com/indexall_big.aspx", 
                          wait_until='networkidle', timeout=30000)
            await asyncio.sleep(1.5)
            
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find main table
            main_table = None
            for table in soup.find_all('table'):
                txt = table.get_text()
                if '時間' in txt and '比賽球隊' in txt:
                    main_table = table
                    break
            
            if not main_table:
                logger.error("❌ Could not find main Titan007 table")
                return []
            
            rows = main_table.find_all('tr')
            logger.info(f"🔍 Found {len(rows)} rows in Titan007 table")
            
            for i, row in enumerate(rows):
                try:
                    if not row.get_text(strip=True):
                        continue
                    
                    # Skip header row
                    if '時間' in row.get_text() and '比賽球隊' in row.get_text():
                        continue
                    
                    # Find team links
                    team1 = row.find('a', id=lambda x: x and x.startswith('team1_'))
                    team2 = row.find('a', id=lambda x: x and x.startswith('team2_'))
                    
                    if not team1 or not team2:
                        continue
                    
                    match_id = team1.get('id', '').replace('team1_', '')
                    
                    # Extract league
                    cells = row.find_all(['td', 'th'])
                    league = "Unknown"
                    if cells:
                        league_text = cells[0].get_text(strip=True)
                        if league_text and league_text != '時間' and '比賽' not in league_text:
                            league = league_text
                    
                    # Extract scheduled time
                    scheduled_time = ""
                    if len(cells) > 1:
                        scheduled_time = cells[1].get_text(strip=True)
                    
                    # Extract status
                    status = ""
                    if len(cells) > 2:
                        status = cells[2].get_text(strip=True)
                    
                    # Clean team names
                    home_team = re.sub(r'\[\d+\]|\(中\)', '', team1.get_text(strip=True)).strip()
                    away_team = re.sub(r'\[\d+\]|\(中\)', '', team2.get_text(strip=True)).strip()
                    
                    if not home_team or not away_team or len(home_team) < 2 or len(away_team) < 2:
                        continue
                    
                    match_dict = {
                        "match_id": match_id,
                        "league": league,
                        "home_team": home_team,
                        "away_team": away_team,
                        "scheduled_time": scheduled_time,
                        "status": status,
                        "scraped_at": datetime.now().isoformat()
                    }
                    
                    matches.append(match_dict)
                    
                except Exception as e:
                    if i < 5:
                        logger.debug(f"Error parsing row {i + 1}: {e}")
                    continue
            
            logger.info(f"✅ Successfully extracted {len(matches)} Titan007 matches")
            return matches
            
        except Exception as e:
            logger.error(f"❌ Error scraping Titan007 match list: {e}")
            return []
        finally:
            await browser.close()


# ==================== Titan Match Stats Scraper ====================

async def scrape_match_stats(titan_match_id: str) -> Dict[str, Any]:
    """
    Scrape detailed statistics for a single Titan007 match.
    
    Args:
        titan_match_id: The Titan007 match ID
        
    Returns:
        Dictionary with match stats, sections, and availability info
    """
    url = f"https://zq.titan007.com/analysis/{titan_match_id}.htm"
    logger.info(f"🔍 Scraping stats for Titan match ID: {titan_match_id}")
    
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
            soup = BeautifulSoup(content, "html.parser")
            
            data = {
                "match_id": titan_match_id,
                "scraped_at": datetime.now().isoformat(),
                "url": url,
                "sections": {},
                "stats_available": False
            }
            
            page_text = soup.get_text(separator=' ', strip=True)
            
            # Check for "no data" patterns
            no_data_patterns = ["暫無數據", "數據統計中", "未開始", "資料準備中", "尚無相關資料", "No data"]
            for pat in no_data_patterns:
                if pat in page_text:
                    logger.warning(f"⚠️ Titan {titan_match_id}: '{pat}' found — no stats available")
                    data["error"] = f"No stats: {pat}"
                    return data
            
            def extract_section_by_regex(regex: str) -> Optional[Any]:
                """Extract a section by regex pattern."""
                header = soup.find(string=re.compile(regex))
                if not header:
                    return None
                
                # Get parent element, handling both Tag and NavigableString
                try:
                    if isinstance(header, Tag):
                        parent = header.find_parent()
                    else:
                        parent = getattr(header, "parent", None)
                except Exception:
                    parent = getattr(header, "parent", None)
                
                if not parent:
                    return None
                
                # Try to extract tables
                tables = parent.find_all('table')
                for table in tables:
                    parsed = extract_table_data(table)
                    if parsed:
                        return parsed
                
                # Fallback to text content
                txt = parent.get_text(separator=' | ', strip=True)
                if txt and len(txt) > 30:
                    return [{"text_content": txt}]
                
                return None
            
            # Extract various sections
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
                        logger.debug(f"✅ Extracted section {key} for match {titan_match_id}")
                except Exception:
                    logger.debug(f"Failed extracting section {key}")
            
            # Extract team formation
            formation_header = soup.find(string=re.compile(r'陣容情況'))
            if formation_header:
                parent = formation_header.find_parent() if hasattr(formation_header, "find_parent") else formation_header.parent
                if parent:
                    data["sections"]["team_formation"] = parent.get_text(separator=' | ', strip=True)
                    sections_found += 1
            
            # Extract ratings
            try:
                home_avg, home_list = None, []
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
                logger.debug(f"Rating extraction exception: {e}")
            
            # Determine if stats are available
            if sections_found >= 1 or data.get("home_rating") or data.get("away_rating"):
                data["stats_available"] = True
                logger.info(f"✅ Titan {titan_match_id}: scraped {sections_found} sections")
            else:
                data["error"] = f"Insufficient stats ({sections_found} sections)"
                logger.warning(f"⚠️ Titan {titan_match_id}: insufficient stats ({sections_found} sections)")
            
            return data
            
        except Exception as e:
            logger.exception(f"Error scraping analysis stats for Titan match {titan_match_id}: {e}")
            return {
                "match_id": titan_match_id,
                "scraped_at": datetime.now().isoformat(),
                "url": url,
                "stats_available": False,
                "error": str(e)
            }
        finally:
            await browser.close()


# ==================== Bulk Scraping ====================

async def scrape_all(max_matches: Optional[int] = None, 
                     output_dir: str = "titan/stats",
                     concurrency: int = 3) -> Dict[str, Any]:
    """
    Scrape all Titan007 matches for today and save to disk.
    
    Args:
        max_matches: Maximum number of matches to scrape (None = all)
        output_dir: Directory to save scraped stats
        concurrency: Number of concurrent scraping tasks
        
    Returns:
        Dictionary with metadata and results
    """
    logger.info("🚀 Starting Titan007 bulk pre-scrape...")
    
    # Step 1: Get match list
    matches = await scrape_titan_match_list()
    if not matches:
        logger.error("No matches found to scrape")
        return {"success": False, "matches_scraped": 0}
    
    if max_matches:
        matches = matches[:max_matches]
    
    logger.info(f"📊 Will scrape {len(matches)} matches...")
    
    # Step 2: Scrape stats for each match
    semaphore = asyncio.Semaphore(concurrency)
    scraped_stats = []
    failed_matches = []
    
    async def scrape_with_semaphore(match):
        async with semaphore:
            try:
                stats = await scrape_match_stats(match["match_id"])
                # Merge match info with stats
                stats["match_info"] = match
                return stats
            except Exception as e:
                logger.error(f"Failed to scrape match {match['match_id']}: {e}")
                failed_matches.append(match["match_id"])
                return None
    
    results = await asyncio.gather(*[scrape_with_semaphore(m) for m in matches])
    scraped_stats = [r for r in results if r is not None]
    
    # Step 3: Save to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    today_str = datetime.now().strftime("%Y%m%d")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual match stats
    for stats in scraped_stats:
        match_id = stats["match_id"]
        file_path = output_path / f"titan_stats_{match_id}_{today_str}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # Save bulk summary
    summary = {
        "scraped_at": timestamp,
        "date": today_str,
        "total_matches": len(matches),
        "scraped_successfully": len(scraped_stats),
        "failed": len(failed_matches),
        "failed_match_ids": failed_matches,
        "stats": scraped_stats
    }
    
    summary_path = output_path / f"titan_stats_bulk_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Bulk scrape complete!")
    logger.info(f"   📊 Scraped: {len(scraped_stats)}/{len(matches)} matches")
    logger.info(f"   💾 Saved to: {output_dir}")
    logger.info(f"   📄 Summary: {summary_path}")
    
    return summary


# ==================== Loading Pre-scraped Data ====================

def load_titan_stats_for_today(stats_dir: str = "titan/stats") -> Dict[str, Dict[str, Any]]:
    """
    Load pre-scraped Titan stats for today from disk.
    
    Args:
        stats_dir: Directory containing scraped stats
        
    Returns:
        Dictionary mapping match_id -> stats
    """
    stats_path = Path(stats_dir)
    if not stats_path.exists():
        logger.info(f"Stats directory {stats_dir} does not exist")
        return {}
    
    today_str = datetime.now().strftime("%Y%m%d")
    stats_map = {}
    
    # Load individual match files
    pattern = f"titan_stats_*_{today_str}.json"
    for file_path in stats_path.glob(pattern):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
                match_id = stats.get("match_id")
                if match_id:
                    stats_map[match_id] = stats
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
    
    logger.info(f"📂 Loaded {len(stats_map)} pre-scraped Titan stats for today")
    return stats_map


def load_titan_stats_from_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load Titan stats from a specific file.
    
    Args:
        file_path: Path to the stats JSON file
        
    Returns:
        Stats dictionary or None if failed
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None


# ==================== CLI ====================

async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Titan007 Pre-Scrape Tool")
    parser.add_argument("--max-matches", type=int, default=None,
                       help="Maximum number of matches to scrape (default: all)")
    parser.add_argument("--output-dir", type=str, default="titan/stats",
                       help="Output directory for scraped stats (default: titan/stats)")
    parser.add_argument("--concurrency", type=int, default=3,
                       help="Number of concurrent scraping tasks (default: 3)")
    
    args = parser.parse_args()
    
    result = await scrape_all(
        max_matches=args.max_matches,
        output_dir=args.output_dir,
        concurrency=args.concurrency
    )
    
    if result.get("scraped_successfully", 0) > 0:
        print(f"\n✅ Success! Scraped {result['scraped_successfully']} matches")
        print(f"📂 Stats saved to: {args.output_dir}")
    else:
        print("\n❌ Scraping failed")


if __name__ == "__main__":
    asyncio.run(main())
