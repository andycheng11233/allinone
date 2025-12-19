# All-in-One Sports Betting Match Analysis Tool

This repository contains a unified script (`all2`) that combines functionality from multiple betting data sources (HKJC, Titan007, MacauSlot) with AI-powered analysis.

## Features

- **HKJC (Hong Kong Jockey Club)**: Scrapes comprehensive odds data including HAD, HDC, HIL, and more
- **Titan007**: Extracts detailed match statistics and analysis
- **MacauSlot**: Gathers live odds from Macau betting platform  
- **AI Analysis**: Uses DeepSeek API to provide betting recommendations based on scraped data
- **Smart Matching**: Intelligently matches games across different platforms
- **Caching**: Avoids re-scraping already processed data

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
playwright install
```

2. Set environment variables:
```bash
export DEEPSEEK_API_KEY=your_api_key_here
export DEEPSEEK_API_URL=https://api.deepseek.com/chat/completions  # optional
```

## Usage

Run the unified script:
```bash
python3 all2
```

Or directly (if executable):
```bash
./all2
```

## Output Files

The script generates several output files:
- `matched_games_with_ai_analysis_*.json` - Matched games with AI recommendations
- `ai_recommendations_*.xlsx` - AI betting recommendations in Excel format
- `all_sources_ordered_*.xlsx` - All data sources combined and organized
- `detailed_report_*.json` - Detailed matching report with metrics
- `hkjc/odds/` - Cached HKJC odds data
- `macauslot/odds/` - Cached MacauSlot odds data

## Configuration

The script uses several environment variables for configuration:
- `DEEPSEEK_API_KEY` - Required for AI analysis
- `DEEPSEEK_API_URL` - API endpoint (default: https://api.deepseek.com/chat/completions)
- `DEEPSEEK_TIMEOUT` - API timeout in seconds (default: 30)
- `DEEPSEEK_RETRIES` - Number of retry attempts (default: 3)

## Notes

- The script automatically caches processed data to avoid redundant scraping
- Cache files are stored in `.cache/` directory
- The `allineone` file is deprecated; use `all2` for all operations