# Changelog

## [Unified] - 2025-12-15

### Added
- Created unified `allinone.py` script combining all functionality from `all2` and `allineone`
- Added `requirements.txt` for dependency management
- Added comprehensive `README.md` with setup and usage instructions
- Added `.gitignore` to exclude cache, build artifacts, and output files
- Added `CHANGELOG.md` to track changes

### Changed
- Consolidated two separate scripts into one unified implementation
- Improved JSON parsing to prevent ReDoS vulnerability (replaced regex with brace-counting algorithm)
- Made script executable with proper shebang

### Removed
- Removed `all2` script (redundant)
- Removed `allineone` script (typo in name, had hardcoded credentials)
- Removed pycache from git tracking

### Security
- Fixed ReDoS (Regular Expression Denial of Service) vulnerability in JSON parsing
- Ensured no hardcoded API keys (all credentials use environment variables)
- All CodeQL security checks passed

### Features
The unified script includes:
- **HKJC Scraper**: Bulk odds scraping with intelligent caching
- **Titan007 Stats**: Comprehensive match statistics extraction
- **MacauSlot Odds**: Live odds from Macau betting platform
- **AI Analysis**: DeepSeek AI integration for betting recommendations
- **Smart Caching**: Reuses cached data to minimize re-scraping
- **Excel Export**: Multiple export formats with matched and unmatched data
- **Multi-source Matching**: Intelligent matching across HKJC, Titan007, and MacauSlot
