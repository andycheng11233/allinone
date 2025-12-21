#!/usr/bin/env python3
"""
Simple test to verify the HKJC event ID scraper integration.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the scraper from all2a
# Note: We'll import the class directly by executing the file
import importlib.util

def load_module_from_file(filepath):
    """Load a Python module from a file path."""
    # Read and compile the file
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Create a module namespace
    import types
    module = types.ModuleType("all2a_module")
    
    # Execute the code in the module's namespace
    exec(code, module.__dict__)
    
    return module

async def test_hkjc_scraper():
    """Test the HKJC home scraper."""
    print("=" * 80)
    print("Testing HKJC Event ID Scraper Integration")
    print("=" * 80)
    
    try:
        # Load the all2a module
        all2a = load_module_from_file("all2a")
        
        # Create scraper instance
        scraper = all2a.HKJCHomeScraper()
        
        print("\n✓ Successfully imported HKJCHomeScraper")
        print(f"✓ Scraper URL: {scraper.urls[0]}")
        print(f"✓ Date format: {scraper.dt_format}")
        print(f"✓ Row selector: {scraper.rows_selector}")
        
        # Test helper methods
        test_date = "21/12/2024 23:00"
        parsed = scraper._parse_row_start(test_date)
        print(f"\n✓ Date parsing test: '{test_date}' → {parsed}")
        
        test_url = "https://bet.hkjc.com/ch/football/allodds/50059049"
        event_id = scraper._extract_id_from_url(test_url)
        print(f"✓ Event ID extraction test: '{test_url}' → {event_id}")
        
        print("\n✅ All basic tests passed!")
        print("\nNote: Full scraping test requires Playwright installation:")
        print("  pip install -r requirements.txt")
        print("  playwright install chromium")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_hkjc_scraper())
    sys.exit(0 if success else 1)
