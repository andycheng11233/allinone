# Quick Start Guide

This guide helps you get started with the refactored match analysis system.

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Playwright Browsers

```bash
playwright install chromium
```

### 3. Set Environment Variables (Optional)

For AI analysis functionality, set your DeepSeek API key:

```bash
export DEEPSEEK_API_KEY="your-api-key-here"
```

Optional configuration:
```bash
export DEEPSEEK_API_URL="https://api.deepseek.com/chat/completions"  # Custom endpoint
export DEEPSEEK_TIMEOUT="30"     # Request timeout in seconds
export DEEPSEEK_RETRIES="3"      # Number of retry attempts
```

## Running the System

### Quick Test

Run the refactored main entry point (currently uses stub implementations):

```bash
python main.py
```

### Running the Original Script

The original monolithic script is still available:

```bash
python A
```

## Project Structure

```
allinone/
├── core/                  # Core functionality
│   ├── utils.py          # Utilities (logging, colors, text processing)
│   ├── alias.py          # Name alias management
│   ├── ai.py             # DeepSeek AI integration
│   └── matcher.py        # Match coordination logic
├── scrapers/              # Web scrapers
│   ├── hkjc.py           # HKJC scraper
│   ├── titan.py          # Titan007 scraper
│   └── macauslot.py      # MacauSlot scraper
├── data/                  # Organized data storage
│   ├── hkjc/             # HKJC data
│   ├── titan/            # Titan007 data
│   └── macauslot/        # MacauSlot data
├── main.py               # New entry point
├── A                      # Original script (for reference)
└── requirements.txt      # Dependencies
```

## Using the Modules

### Example: Using Core Utilities

```python
from core import cprint, Fore, normalize_team_name

# Print colored output
cprint("Hello World!", Fore.GREEN)

# Normalize team names
normalized = normalize_team_name("Real Madrid CF")
print(f"Normalized: {normalized}")  # Output: real madrid
```

### Example: Using Alias Management

```python
from core.alias import normalize_team_name, upsert_alias, save_alias_table_if_needed

# Add a new alias
upsert_alias("teams", "Real Madrid Club de Fútbol", "hkjc")
upsert_alias("teams", "Real Madrid", "titan")

# Save changes
save_alias_table_if_needed()

# Resolve to canonical form
canonical = normalize_team_name("Real Madrid CF")
```

### Example: Using AI Functions

```python
import asyncio
from core.ai import call_deepseek_api, normalize_parsed_data

async def analyze():
    # Prepare data
    stats = {
        "match": {"home_team": "Team A", "away_team": "Team B"},
        "sections": {"recent10_ratings_parsed": {...}}
    }
    
    # Normalize
    normalized = normalize_parsed_data(stats)
    
    # Call AI
    response = await call_deepseek_api("Your prompt here")
    print(response)

asyncio.run(analyze())
```

### Example: Using Scrapers

```python
import asyncio
from scrapers.hkjc import HKJCHomeScraper

async def scrape_hkjc():
    scraper = HKJCHomeScraper()
    matches = await scraper.scrape()
    
    for match in matches:
        print(f"{match['home_team']} vs {match['away_team']}")

asyncio.run(scrape_hkjc())
```

## Current Status

### ✅ Complete
- Core utilities module
- Alias management module
- AI integration module
- Module structure and imports
- Documentation
- Basic testing

### 🚧 In Progress (Stub Implementation)
- HKJC scrapers - Structure created, parsing functions need completion
- Titan007 scraper - Structure created, implementation needed
- MacauSlot scraper - Structure created, implementation needed
- LiveMatchMatcher - Structure created, full logic needs extraction

### 📝 TODO
- Complete extraction of all code from script A
- Add unit tests for each module
- Add integration tests
- Performance optimization
- Database migration support

## Development Workflow

### 1. Make Changes
Edit files in `core/` or `scrapers/` directories.

### 2. Test Imports
```bash
python -c "from core import cprint, Fore; cprint('Test', Fore.GREEN)"
```

### 3. Run Tests
```bash
# Run individual module tests
python core/matcher.py
python scrapers/hkjc.py

# Run main application
python main.py
```

### 4. Verify Against Original
```bash
# Compare output with original script
python A
```

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError`:
```bash
pip install -r requirements.txt
```

### Playwright Errors
If browser automation fails:
```bash
playwright install chromium
```

### API Errors
If AI analysis fails, check your API key:
```bash
echo $DEEPSEEK_API_KEY
```

## Data Organization

The system saves data in organized directories:

```
data/
├── hkjc/
│   ├── odds/              # HKJC odds files (JSON)
│   └── matches/           # HKJC match listings
├── titan/
│   └── stats/             # Titan statistics
│       ├── full/          # Complete datasets
│       ├── incomplete/    # Partial datasets
│       └── missing/       # Missing data markers
└── macauslot/
    └── odds/              # MacauSlot odds files
```

This structure makes it easy to:
- Back up specific data sources
- Migrate to a database later
- Debug and analyze data
- Version control data (if needed)

## Next Steps

1. Review the `README_REFACTOR.md` for detailed architecture information
2. Explore individual module files to understand the structure
3. Check TODO comments in the code for areas needing completion
4. Run `python main.py` to see the current stub implementation in action
5. Contribute to completing the extraction from script A

## Getting Help

- Check `README_REFACTOR.md` for architecture details
- Look at inline documentation in module files
- Review TODO comments for guidance on what needs to be done
- Compare with original script A for reference

## Contributing

When adding new code:
1. Follow the existing module structure
2. Add docstrings to all functions and classes
3. Update relevant README files
4. Test imports and basic functionality
5. Mark incomplete sections with TODO comments

Happy coding! 🚀
