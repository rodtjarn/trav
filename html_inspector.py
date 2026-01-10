#!/usr/bin/env python3
"""
HTML Structure Inspector for ATG.se
This tool helps you identify the correct CSS selectors and HTML structure
"""

import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime


def inspect_atg_structure(url: str):
    """
    Inspect and display the HTML structure of an ATG page
    
    Args:
        url: URL to inspect
    """
    print(f"\n{'='*80}")
    print(f"Inspecting: {url}")
    print(f"{'='*80}\n")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all unique class names
        print("UNIQUE CLASS NAMES FOUND:")
        print("-" * 80)
        classes = set()
        for element in soup.find_all(class_=True):
            if isinstance(element.get('class'), list):
                classes.update(element.get('class'))
        
        relevant_classes = [c for c in sorted(classes) if any(keyword in c.lower() 
            for keyword in ['race', 'horse', 'start', 'runner', 'häst', 'lopp', 
                          'program', 'kusk', 'driver', 'odds', 'spel'])]
        
        for cls in relevant_classes[:50]:  # Show first 50 relevant classes
            print(f"  .{cls}")
        
        print(f"\nTotal unique classes: {len(classes)}")
        print(f"Relevant classes shown: {min(50, len(relevant_classes))}")
        
        # Find all unique IDs
        print("\n" + "="*80)
        print("UNIQUE IDs FOUND:")
        print("-" * 80)
        ids = set()
        for element in soup.find_all(id=True):
            ids.add(element.get('id'))
        
        relevant_ids = [i for i in sorted(ids) if any(keyword in i.lower() 
            for keyword in ['race', 'horse', 'start', 'runner', 'program'])]
        
        for id_name in relevant_ids[:30]:
            print(f"  #{id_name}")
        
        # Find common data attributes
        print("\n" + "="*80)
        print("DATA ATTRIBUTES FOUND:")
        print("-" * 80)
        data_attrs = set()
        for element in soup.find_all():
            for attr in element.attrs:
                if attr.startswith('data-'):
                    data_attrs.add(attr)
        
        for attr in sorted(data_attrs)[:30]:
            print(f"  {attr}")
        
        # Look for JSON/script data
        print("\n" + "="*80)
        print("SCRIPT TAGS WITH JSON DATA:")
        print("-" * 80)
        scripts = soup.find_all('script', type='application/json')
        for i, script in enumerate(scripts[:5]):
            print(f"\nScript {i+1}:")
            try:
                data = json.loads(script.string)
                print(f"  Keys: {list(data.keys())[:10]}")
            except:
                print(f"  Content preview: {script.string[:200]}...")
        
        # Save full HTML for manual inspection
        output_file = f"atg_structure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(soup.prettify())
        print(f"\n\nFull HTML saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")


def find_race_data_structure(url: str):
    """
    Attempt to find and display the structure of race data
    """
    print(f"\n{'='*80}")
    print("SEARCHING FOR RACE DATA STRUCTURES")
    print(f"{'='*80}\n")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for table structures
        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables")
        
        for i, table in enumerate(tables[:3]):
            print(f"\nTable {i+1} structure:")
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            print(f"  Headers: {headers}")
            
            rows = table.find_all('tr')[:3]
            print(f"  Sample rows: {len(rows)}")
            for j, row in enumerate(rows):
                cells = [td.get_text(strip=True)[:30] for td in row.find_all(['td', 'th'])]
                print(f"    Row {j}: {cells}")
        
        # Look for list structures
        print("\n" + "="*80)
        print("LOOKING FOR LIST STRUCTURES:")
        print("-" * 80)
        
        for tag in ['ul', 'ol']:
            lists = soup.find_all(tag, class_=lambda x: x and any(
                keyword in str(x).lower() for keyword in ['race', 'horse', 'start', 'runner']))
            print(f"\nFound {len(lists)} relevant <{tag}> elements")
            
            for i, list_elem in enumerate(lists[:2]):
                print(f"\n{tag.upper()} {i+1}:")
                print(f"  Classes: {list_elem.get('class')}")
                items = list_elem.find_all('li')[:3]
                print(f"  Items: {len(items)}")
                for j, item in enumerate(items):
                    print(f"    Item {j}: {item.get_text(strip=True)[:50]}")
        
    except Exception as e:
        print(f"Error: {e}")


def check_api_endpoints():
    """
    Check for common API endpoints
    """
    print(f"\n{'='*80}")
    print("CHECKING FOR API ENDPOINTS")
    print(f"{'='*80}\n")
    
    base_urls = [
        "https://www.atg.se/api",
        "https://www.atg.se/services/racinginfo/v1",
        "https://api.atg.se",
        "https://www.travsport.se/api",
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for url in base_urls:
        try:
            response = requests.get(url, headers=headers, timeout=5)
            print(f"✓ {url}")
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                print(f"  Content preview: {response.text[:200]}")
        except Exception as e:
            print(f"✗ {url}")
            print(f"  Error: {str(e)[:100]}")
        print()


if __name__ == "__main__":
    import sys
    
    # Check if URL provided as argument
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        # Default to today's program
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"https://www.atg.se/spel/kalender/{today}"
    
    print("ATG.se Structure Inspector")
    print("=" * 80)
    
    # Run inspections
    inspect_atg_structure(url)
    find_race_data_structure(url)
    check_api_endpoints()
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("1. Review the saved HTML file to understand the structure")
    print("2. Update the CSS selectors in atg_scraper.py based on findings")
    print("3. Test with a small date range first")
    print("="*80)
