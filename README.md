# allinone

Football betting analysis toolkit combining HKJC, Titan007, and MacauSlot data sources with AI-powered recommendations.

## Features

- **HKJC Scraper**: Validated event ID scraper with interactive browser automation
  - Loads bet.hkjc.com/ch/football/home
  - Clicks "顯示更多" to reveal all matches
  - Skips already-started games based on date parsing
  - Uses Ctrl/Cmd-click to capture event IDs from odds pages
  - Monitors network requests for /allodds/ URLs
  
- **Titan007 Integration**: Match statistics and analysis data
- **MacauSlot Odds**: Live betting odds collection
- **AI Analysis**: DeepSeek-powered betting recommendations
- **Multi-source Matching**: Intelligent match correlation across platforms

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Playwright browser
playwright install chromium
```

### Usage

```bash
# Run the main script
python3 all2a

# Verify the HKJC scraper integration
python3 verify_integration.py
```

### Environment Variables

- `DEEPSEEK_API_KEY`: API key for DeepSeek AI analysis (optional)
- `DEEPSEEK_API_URL`: DeepSeek API endpoint (default: https://api.deepseek.com/chat/completions)

## Documentation

- [INTEGRATION.md](INTEGRATION.md) - Detailed documentation of the HKJC scraper integration

## Scripts

- **all2a**: Main unified scraper and analysis pipeline
- **hkjc_event**: Standalone HKJC event ID scraper (validation reference)
- **verify_integration.py**: Verify HKJC scraper integration
- **test_scraper.py**: Basic test script

## Output

The script generates several output files:

- `hkjc/odds/` - HKJC odds data
- `titan/stats/` - Titan007 match statistics  
- `macau/` - MacauSlot odds
- `*.xlsx` - Excel reports with matched data and AI recommendations
- `*.json` - Raw JSON data exports

## Integration Details

See [INTEGRATION.md](INTEGRATION.md) for complete details on the HKJC event ID scraper integration, including:

- Architecture and design
- Configuration options
- Usage examples
- Troubleshooting guide
- Comparison with legacy implementation