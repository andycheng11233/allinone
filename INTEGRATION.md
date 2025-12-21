# HKJC Event ID Scraper Integration

## Overview

This document describes the integration of the validated HKJC event ID scraper into the `all2a` script. The scraper has been successfully integrated as a reusable class within the existing codebase.

## What Changed

### Modified Files

- **all2a**: Replaced the legacy `HKJCHomeScraper` class with the validated implementation

### New Files

- **.gitignore**: Excludes Python cache, build artifacts, and output files
- **requirements.txt**: Lists required Python dependencies
- **test_scraper.py**: Basic test script for the scraper
- **verify_integration.py**: Verification tool to validate the integration

## Features

The integrated `HKJCHomeScraper` class now includes the following validated features:

### 1. Interactive Event ID Collection
- Loads https://bet.hkjc.com/ch/football/home
- Uses Playwright browser automation for reliable scraping

### 2. Smart Content Loading
- Repeatedly clicks "顯示更多" (Show More) button
- Scrolls to load lazy-loaded content
- Continues until row count stops growing or declared count is reached

### 3. Already-Started Game Filtering
- Parses match start times in format `%d/%m/%Y %H:%M` (e.g., "21/12/2024 23:00")
- Skips games where start time is before current time
- Prevents wasted processing of completed matches

### 4. Multiple Trigger Selection
The scraper tries multiple selectors per row to find clickable elements:
1. `[title*="賠率"]` - Elements with "賠率" (odds) in title
2. `[title*="所有賠率"]` - Elements with "所有賠率" (all odds) in title
3. `.teamIconSmall [title]` - Team icon elements with titles
4. `.teamIconSmall` - Team icon elements
5. `.team` - Team name elements

### 5. Keyboard-Modified Clicks
- Uses Ctrl-click (Windows/Linux) or Cmd-click (macOS) to open links
- Opens odds pages in new tabs/windows without losing place
- Detects the platform automatically

### 6. Dual URL Capture Method
- **Network monitoring**: Captures `/allodds/` URLs from browser network requests/responses
- **Popup detection**: Detects new tabs/windows and extracts URLs
- **Navigation detection**: Detects same-page navigation and extracts URLs

### 7. Robust Error Handling
- Continues processing even if individual rows fail
- Logs detailed debug information for troubleshooting
- Gracefully handles missing elements and timeouts

## Usage

### Within all2a Script

The scraper is used internally by the `all2a` script as part of its bulk odds collection:

```python
# Create scraper instance
home_scraper = HKJCHomeScraper()

# Scrape event IDs
home_rows = await home_scraper.scrape()

# Results contain event IDs and basic match info
for row in home_rows:
    event_id = row["event_id"]
    # Process event ID...
```

### Standalone Usage

The scraper can also be used independently:

```python
import asyncio
from all2a import HKJCHomeScraper

async def main():
    scraper = HKJCHomeScraper()
    results = await scraper.scrape()
    
    for match in results:
        print(f"Event ID: {match['event_id']}")

asyncio.run(main())
```

## Configuration

The scraper exposes several configuration parameters:

```python
scraper = HKJCHomeScraper()

# Configurable attributes (set after initialization)
scraper.dt_format = "%d/%m/%Y %H:%M"  # Date format for parsing
scraper.rows_selector = ".match-row,.event-row"  # CSS selector for match rows
scraper.pause = 0.6  # Pause between scroll actions (seconds)
scraper.click_wait = 2.0  # Wait after clicking (seconds)
scraper.timeout_ms = 12000  # Timeout for popup detection (milliseconds)
```

## Dependencies

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

For Playwright browser automation, also install the browser:

```bash
playwright install chromium
```

### Required Packages

- **playwright>=1.40.0**: Browser automation
- **beautifulsoup4>=4.12.0**: HTML parsing (used elsewhere in all2a)
- **httpx>=0.25.0**: Async HTTP client (used elsewhere in all2a)
- **pandas>=2.1.0**: Data manipulation (used elsewhere in all2a)
- **openpyxl>=3.1.0**: Excel file support (used elsewhere in all2a)
- **colorama>=0.4.6**: Colored console output (used elsewhere in all2a)

## Verification

Run the verification script to confirm the integration:

```bash
python3 verify_integration.py
```

Expected output:
```
================================================================================
Verifying HKJC Event ID Scraper Integration
================================================================================

✓ Syntax is valid
✓ HKJCHomeScraper class found
✓ Found 8 methods
✓ Class has docstring
✓ Found 5/5 key features mentioned

...

✅ Integration verification completed successfully!
```

## Architecture

### Class Structure

```
HKJCHomeScraper
├── __init__(urls=None)
├── _parse_row_start(txt: str) -> Optional[datetime]
├── _extract_id_from_url(url: str) -> Optional[str]
├── _get_declared_count(page) -> Optional[int]
├── _scroll_bottom(page)
├── _click_show_more_once(page) -> bool
├── _scrape_one(url: str) -> List[Dict[str, Any]]
└── scrape() -> List[Dict[str, Any]]
```

### Data Flow

1. **Initialize**: Create scraper with target URLs
2. **Navigate**: Load HKJC home page with Playwright
3. **Monitor**: Set up network request/response listeners
4. **Expand**: Click "Show More" and scroll to reveal all matches
5. **Iterate**: Process each match row
6. **Filter**: Skip already-started games based on date
7. **Interact**: Try multiple triggers with keyboard-modified clicks
8. **Capture**: Extract event IDs from network traffic and popups
9. **Return**: Deduplicated list of event IDs

### Return Format

```python
[
    {
        "event_id": "50059049",
        "home_team": None,  # Not extracted in this implementation
        "away_team": None,  # Not extracted in this implementation
        "raw_text": ""
    },
    # ... more matches
]
```

## Comparison with Legacy Implementation

| Feature | Legacy | Validated (New) |
|---------|--------|-----------------|
| Method | Static HTML parsing | Interactive browser automation |
| URL Capture | Parse static content | Monitor network + detect popups |
| Started Game Filtering | ❌ None | ✓ Date-based filtering |
| Content Loading | Scroll only | Click "Show More" + scroll |
| Trigger Selection | Single method | Multiple fallback selectors |
| Reliability | Medium | High |
| Network Monitoring | ❌ | ✓ |
| Popup Detection | ❌ | ✓ |

## Known Limitations

1. **Team Names**: Current implementation focuses on event ID collection only. Team names are set to `None` and can be enriched later from other sources.

2. **Headless Mode**: Runs in headless mode by default. Some sites may detect this, though HKJC appears to work fine.

3. **Platform Dependency**: Requires Playwright and Chromium browser to be installed.

4. **Performance**: Interactive scraping is slower than static parsing but more reliable.

## Troubleshooting

### Issue: Browser Not Installed

**Error**: `playwright._impl._api_types.Error: Executable doesn't exist`

**Solution**:
```bash
playwright install chromium
```

### Issue: Module Not Found

**Error**: `ModuleNotFoundError: No module named 'playwright'`

**Solution**:
```bash
pip install -r requirements.txt
```

### Issue: No Event IDs Collected

**Possible causes**:
1. HKJC website structure changed - update selectors
2. All games already started - normal behavior
3. Network issues - check internet connection
4. Timeout too short - increase `timeout_ms`

**Debug**:
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Potential improvements for future versions:

1. **Team Name Extraction**: Parse team names directly from rows during scraping
2. **Match Time Extraction**: Extract and store match times with event IDs
3. **Tournament Info**: Capture tournament/league information
4. **Retry Logic**: Add automatic retries for failed captures
5. **Rate Limiting**: Implement delays to avoid overwhelming the server
6. **Caching**: Cache results to avoid re-scraping recent events
7. **Parallel Processing**: Process multiple rows concurrently for speed

## Testing

### Manual Testing

To manually test the scraper:

```bash
# Test with the standalone hkjc_event script first
python3 hkjc_event

# Then test the integration in all2a
python3 all2a
```

### Automated Testing

The `verify_integration.py` script performs static analysis without running the scraper:

```bash
python3 verify_integration.py
```

For full integration testing, ensure all dependencies are installed and run the full `all2a` script.

## Credits

- Original validated scraper: `hkjc_event`
- Integration: all2a script
- Platform: HKJC (Hong Kong Jockey Club) betting platform

## License

This code is part of the allinone repository and follows the same license.
