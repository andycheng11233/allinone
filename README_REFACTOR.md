# Refactored Project Structure

This document explains the modular structure created from the monolithic script `A`.

## Directory Structure

```
allinone/
├── core/                      # Core functionality modules
│   ├── __init__.py           # Package initialization
│   ├── utils.py              # Shared utilities (logging, colors, text processing)
│   ├── alias.py              # Team/league name alias management
│   └── ai.py                 # DeepSeek AI integration
├── scrapers/                  # Web scraping modules (to be populated)
│   ├── __init__.py           # Package initialization
│   ├── hkjc.py               # HKJC scrapers (planned)
│   ├── titan.py              # Titan007 scraper (planned)
│   └── macauslot.py          # MacauSlot scraper (planned)
├── data/                      # Organized data storage
│   ├── hkjc/                 # HKJC data files
│   ├── titan/                # Titan007 data files
│   └── macauslot/            # MacauSlot data files
├── A                          # Original monolithic script (kept for reference)
├── requirements.txt           # Python dependencies
└── README_REFACTOR.md        # This file
```

## Module Organization

### Core Modules

#### `core/utils.py`
Contains shared utilities used throughout the application:
- **Logging**: Configured logging with proper levels
- **Color printing**: `cprint()` function for colored terminal output
- **Text processing**: 
  - `strip_accents()`: Remove accents from text
  - `find_best_float_in_text()`: Extract float values from text
  - `parse_decimal_tokens_from_concatenated()`: Parse decimal values
  - `extract_ratings_or_average_from_text()`: Extract ratings from page text
- **Constants**: Shared constants like cache paths, Titan stats paths, etc.

#### `core/alias.py`
Manages team and league name aliases across different sources:
- **Alias loading/saving**: 
  - `load_alias_table_from_json()`: Load aliases from JSON
  - `save_alias_table_if_needed()`: Save if modified
- **Name resolution**:
  - `resolve_alias()`: Resolve team names to canonical form
  - `resolve_league_alias()`: Resolve league names
- **Name normalization**:
  - `normalize_team_name()`: Normalize team names for comparison
  - `normalize_league()`: Normalize league names
- **Alias management**:
  - `upsert_alias()`: Add or update alias entries
  - `append_unalias_pending()`: Track unresolved names
- **Canonical preference**: HKJC > MacauSlot > Titan

#### `core/ai.py`
Integrates with DeepSeek AI for match analysis:
- **API communication**:
  - `call_deepseek_api()`: Make API calls with retry logic
- **Data preparation**:
  - `normalize_parsed_data()`: Prepare data for AI analysis
  - `has_meaningful_data_for_ai()`: Check if data is sufficient
  - `build_ai_prompt_with_availability()`: Build AI prompts
- **Response handling**:
  - `parse_ai_json_response()`: Parse AI responses
  - `perform_ai_analysis_for_match()`: Complete AI analysis workflow

### Scraper Modules (Planned)

#### `scrapers/hkjc.py` (To be implemented)
Will contain:
- `HKJCHomeScraper`: Scrape HKJC home page for matches
- `HKJCDetailedOddsScraper`: Scrape detailed odds for events
- `HKJCBulkOddsCollector`: Collect odds in bulk
- All HKJC parsing functions (`ao_parse_*`)

#### `scrapers/titan.py` (To be implemented)
Will contain:
- `TitanStatsScraper`: Scrape Titan007 statistics
- Titan-specific parsing functions

#### `scrapers/macauslot.py` (To be implemented)
Will contain:
- `MacauSlotOddsScraper`: Scrape MacauSlot odds

### Main Entry Point (Planned)

#### `main.py` (To be implemented)
Will orchestrate the entire workflow:
1. Initialize scrapers
2. Collect data from all sources
3. Match games across platforms
4. Run AI analysis
5. Generate reports

## Data Organization

The `data/` directory provides structured storage for scraped data:

```
data/
├── hkjc/
│   ├── odds/              # HKJC odds files
│   └── matches/           # HKJC match data
├── titan/
│   └── stats/             # Titan statistics
│       ├── full/          # Complete stats
│       ├── incomplete/    # Partial stats
│       └── missing/       # Missing stats
└── macauslot/
    └── odds/              # MacauSlot odds files
```

This structure facilitates:
- Easy backup and versioning
- Future database migration
- Data analysis and debugging

## Dependencies

Listed in `requirements.txt`:
- `playwright`: Browser automation for scraping
- `pandas`: Data manipulation and Excel export
- `beautifulsoup4`: HTML parsing
- `httpx`: Async HTTP client
- `colorama`: Colored terminal output
- `openpyxl`: Excel file support

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

## Usage

### Current State
The original monolithic script `A` is still functional and can be run as before:
```bash
python A
```

### Future Usage (After complete refactoring)
```bash
# Run the refactored version
python main.py
```

## Benefits of This Structure

1. **Maintainability**: Each module has a clear, focused responsibility
2. **Testability**: Individual components can be tested in isolation
3. **Reusability**: Core functions can be imported and reused
4. **Debuggability**: Easier to locate and fix issues
5. **Collaboration**: Multiple developers can work on different modules
6. **Future-Ready**: Prepared for database migration and scaling

## Migration Notes

### From Original Script to Modules

When migrating code, follow these patterns:

**Imports in new modules:**
```python
# Instead of having everything in one file
# Import what you need from other modules
from core.utils import cprint, logger, Fore
from core.alias import normalize_team_name, upsert_alias
from core.ai import call_deepseek_api, perform_ai_analysis_for_match
```

**Using the modules:**
```python
# In your code, use the imported functions
from core import cprint, Fore, normalize_team_name

def my_function():
    cprint("Processing match data...", Fore.CYAN)
    normalized_name = normalize_team_name("Real Madrid CF")
```

## Next Steps

1. Complete `scrapers/hkjc.py` with all HKJC scraping logic
2. Complete `scrapers/titan.py` with Titan007 scraping
3. Complete `scrapers/macauslot.py` with MacauSlot scraping
4. Extract `LiveMatchMatcher` class to `core/matcher.py`
5. Create `main.py` as the new entry point
6. Add unit tests for each module
7. Update documentation with examples

## Backward Compatibility

The original script `A` remains in the repository for:
- Reference during refactoring
- Fallback if issues arise
- Comparison testing

Once the refactored version is stable and tested, script `A` can be moved to an archive directory or removed.
