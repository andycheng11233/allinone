# GitHub Copilot Instructions

## Project Overview

This is a unified sports betting data scraper and analyzer that aggregates odds and statistics from multiple sources (MacauSlot, Titan007, and HKJC - Hong Kong Jockey Club). The project includes AI-powered analysis using the DeepSeek API and exports results to Excel.

## Key Technologies

- **Python 3**: Primary programming language
- **Playwright**: Browser automation for web scraping
- **BeautifulSoup**: HTML parsing
- **Pandas**: Data manipulation and Excel export
- **httpx**: Async HTTP client
- **DeepSeek API**: AI analysis integration

## Project Structure

- `all2`: Main unified scraper with AI analysis
- `allineone`: Alternative/backup version of the scraper
- `.cache/`: Directory for caching processed data (AI results, HKJC odds, Titan stats)
- Output directories: `hkjc/odds/`, `macau/`, `titan/`, `comparison/`, `ai/`

## Coding Standards

### Python Style
- Use type hints for function parameters and return values
- Follow PEP 8 conventions
- Use descriptive variable names
- Add docstrings to complex functions
- Handle exceptions gracefully with proper error logging

### Async Programming
- Use `async`/`await` for I/O operations
- Properly manage Playwright browser contexts and pages
- Close resources in `finally` blocks or use context managers

### Data Handling
- Cache processed data to avoid redundant API calls and scraping
- Use Path objects for file system operations
- Ensure all JSON files use UTF-8 encoding with `ensure_ascii=False`
- Validate data types before processing (e.g., check for None, empty strings)

### Web Scraping Best Practices
- Add appropriate delays between requests to be respectful
- Handle missing elements gracefully (elements may not always exist)
- Use multiple selectors as fallbacks when extracting data
- Log scraping progress and errors clearly

### AI Integration
- Check for `DEEPSEEK_API_KEY` environment variable before making API calls
- Implement retry logic for API failures
- Cache AI results to avoid redundant processing
- Include timeout handling for API requests

## Environment Variables

Required:
- `DEEPSEEK_API_KEY`: API key for DeepSeek AI service

Optional:
- `DEEPSEEK_API_URL`: Custom API endpoint (default: https://api.deepseek.com/chat/completions)
- `DEEPSEEK_TIMEOUT`: Request timeout in seconds (default: 30)
- `DEEPSEEK_RETRIES`: Number of retry attempts (default: 3)

## Setup Instructions

```bash
pip install -r requirements.txt
playwright install
export DEEPSEEK_API_KEY=your_api_key_here
export DEEPSEEK_API_URL=https://api.deepseek.com/chat/completions
```

## Running the Application

```bash
python3 all2
# or
python3 allineone
```

## Testing Considerations

- Test scraping logic with mock HTML responses when possible
- Verify cache loading/saving functionality
- Test data matching algorithms with sample data
- Validate Excel export format and content
- Check error handling for missing data sources

## Important Context

### Data Sources
1. **HKJC (Hong Kong Jockey Club)**: Primary odds provider
2. **Titan007**: Statistics provider
3. **MacauSlot**: Additional odds source

### Matching Logic
- Uses fuzzy matching (`difflib.SequenceMatcher`) to match events across different sources
- Implements 1:1 matching to avoid Cartesian product explosion
- Supports caching to skip already processed event IDs

### Output Format
- Excel files with matched rows first, followed by unmatched rows from each source
- JSON files for caching and debugging
- HTML snapshots for debugging scraper issues

## Common Pitfalls

- Always check if elements exist before accessing them in web scraping
- Handle different text encodings (Chinese characters are common)
- Browser automation may be flaky; implement retries
- API rate limiting should be considered
- Cache files may become corrupted; validate JSON before parsing

## When Making Changes

1. **Maintain backward compatibility** with existing cache files
2. **Test thoroughly** with real data sources when possible
3. **Update cache schemas carefully** to avoid breaking existing functionality
4. **Log important events** for debugging purposes
5. **Handle edge cases** like missing data, network errors, and parsing failures
6. **Document any new environment variables** or configuration options
