# All-in-One Football Match Analysis

A unified Python script that combines data from MacauSlot, Titan007, and HKJC to provide comprehensive football match analysis with AI-powered betting recommendations.

## Features

- **Multi-Source Data Integration**: Scrapes and matches data from HKJC, Titan007, and MacauSlot
- **HKJC All-Odds Scraper**: Bulk scrapes detailed odds from HKJC with intelligent caching
- **Titan007 Stats**: Extracts comprehensive match statistics and analysis
- **MacauSlot Odds**: Live odds from Macau betting platform
- **AI Analysis**: DeepSeek AI integration for betting recommendations
- **Smart Caching**: Reuses cached data across runs to minimize re-scraping
- **Excel Export**: Multiple export formats with matched and unmatched data

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
playwright install
```

2. Set up environment variables:
```bash
export DEEPSEEK_API_KEY=your_api_key_here
export DEEPSEEK_API_URL=https://api.deepseek.com/chat/completions
```

## Usage

Run the unified script:
```bash
python3 allinone.py
```

## Output Files

The script generates several output files:
- `hkjc/odds/hkjc_allodds_<timestamp>.json` - HKJC detailed odds
- `macauslot/odds/macauslot_odds_<timestamp>.json` - MacauSlot odds
- `matched_games_with_ai_analysis_<timestamp>.json` - Matched games with AI analysis
- `ai_recommendations_<timestamp>.xlsx` - AI betting recommendations
- `all_sources_ordered_<timestamp>.xlsx` - Combined data from all sources
- `detailed_report_<timestamp>.json` - Analysis report

## Cache Management

The script maintains cache files in `.cache/` directory:
- `ai_processed.json` - Processed AI analyses
- `hkjc_odds_processed.json` - Processed HKJC event IDs
- `titan_stats_processed.json` - Processed Titan007 match IDs

## Configuration

Environment variables:
- `DEEPSEEK_API_KEY` - Required for AI analysis
- `DEEPSEEK_API_URL` - DeepSeek API endpoint (default: https://api.deepseek.com/chat/completions)
- `DEEPSEEK_TIMEOUT` - API timeout in seconds (default: 30)
- `DEEPSEEK_RETRIES` - Number of retries for failed API calls (default: 3)