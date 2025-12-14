# allinone

Unified MacauSlot + Titan007 + HKJC matcher with AI analysis.

## Features

- **HKJC All-Odds Scraping**: Bulk scrape and cache HKJC odds
- **Titan007 Statistics**: Scrape detailed match statistics from Titan007
- **MacauSlot Odds**: Collect odds from MacauSlot
- **AI Analysis**: Integrated DeepSeek AI for betting recommendations
- **Titan Pre-Scrape Workflow**: Bulk scrape Titan stats in advance to reduce runtime

## Titan Pre-Scrape Workflow

The Titan pre-scrape workflow allows you to bulk-scrape Titan007 statistics in advance, which are then reused by the main matcher. This significantly reduces the runtime of the all-in-one matcher.

### Usage

#### 1. Pre-scrape Titan stats (run once per day)

```bash
# Scrape all matches for today
python3 titan_pre_scrape.py

# Limit to first 50 matches
python3 titan_pre_scrape.py --max-matches 50

# Specify custom output directory
python3 titan_pre_scrape.py --output-dir /path/to/titan/stats
```

The scraped stats are saved to `titan/stats/` by default, with files named:
- `titan_stats_{match_id}_{YYYYMMDD}.json` - Individual match stats
- `titan_stats_bulk_{YYYYMMDD_HHMMSS}.json` - Bulk summary

#### 2. Run the all-in-one matcher

```bash
python3 allineone
```

The matcher will automatically detect and use pre-scraped Titan stats for today. If stats are not available for a match, it will fall back to live scraping.

### Python API

```python
from titan_pre_scrape import scrape_all, load_titan_stats_for_today

# Scrape all matches and save to disk
await scrape_all(max_matches=50, output_dir="titan/stats")

# Load pre-scraped stats for today
stats = load_titan_stats_for_today(stats_dir="titan/stats")
```

### Benefits

- **Faster execution**: Reuse scraped data instead of re-scraping every match
- **Reduced load**: Minimize requests to Titan007 servers
- **Scheduled pre-scraping**: Run pre-scrape on a schedule (e.g., cron job)
- **Graceful fallback**: Works without pre-scraped data (falls back to live scraping)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install

# Set environment variables
export DEEPSEEK_API_KEY=your_api_key_here
export DEEPSEEK_API_URL=https://api.deepseek.com/chat/completions
```

## Run

```bash
# Main matcher
python3 allineone

# Pre-scrape workflow
python3 titan_pre_scrape.py
```