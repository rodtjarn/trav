#!/usr/bin/env python3
"""
Swedish Trotting Data Scraper
Collects race program data from ATG.se and travsport.se

IMPORTANT: 
- Check robots.txt and terms of service before using
- Use responsibly with rate limiting
- For research/educational purposes only
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SwedishTrottingScraper:
    """Scraper for Swedish trotting race data"""
    
    def __init__(self, delay: float = 2.0):
        """
        Initialize scraper
        
        Args:
            delay: Delay between requests in seconds (be respectful!)
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def _sleep(self):
        """Polite delay between requests"""
        time.sleep(self.delay)
        
    def get_race_days(self, days_ahead: int = 7) -> List[str]:
        """
        Get upcoming race dates
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of date strings in YYYY-MM-DD format
        """
        dates = []
        today = datetime.now()
        for i in range(days_ahead):
            date = today + timedelta(days=i)
            dates.append(date.strftime('%Y-%m-%d'))
        return dates
    
    def scrape_atg_program(self, date: str) -> List[Dict]:
        """
        Scrape race program from ATG.se for a specific date
        
        Args:
            date: Date string in YYYY-MM-DD format
            
        Returns:
            List of race data dictionaries
        """
        races_data = []
        
        try:
            # ATG's kalender/program URL structure
            url = f"https://www.atg.se/spel/kalender/{date}"
            logger.info(f"Fetching ATG program for {date}")
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find race meetings/tracks
            # Note: This is a template - actual selectors need to be updated based on current ATG HTML
            race_cards = soup.find_all('div', class_=re.compile(r'race.*card|program.*item', re.I))
            
            for card in race_cards:
                race_info = self._parse_atg_race_card(card)
                if race_info:
                    races_data.append(race_info)
            
            logger.info(f"Found {len(races_data)} races for {date}")
            
        except Exception as e:
            logger.error(f"Error scraping ATG program for {date}: {e}")
        
        self._sleep()
        return races_data
    
    def _parse_atg_race_card(self, card_element) -> Optional[Dict]:
        """Parse individual race card from ATG"""
        try:
            # Template parsing - adjust selectors based on actual HTML structure
            race_data = {
                'track': card_element.find('span', class_=re.compile('track|venue')),
                'race_number': card_element.find('span', class_=re.compile('race.*number')),
                'start_time': card_element.find('time', class_=re.compile('time|start')),
                'race_type': card_element.find('span', class_=re.compile('type|category')),
            }
            
            # Extract text from elements
            for key, element in race_data.items():
                if element:
                    race_data[key] = element.get_text(strip=True)
                else:
                    race_data[key] = None
                    
            return race_data if any(race_data.values()) else None
            
        except Exception as e:
            logger.error(f"Error parsing race card: {e}")
            return None
    
    def scrape_race_details(self, race_id: str) -> Dict:
        """
        Scrape detailed information for a specific race
        
        Args:
            race_id: Race identifier from ATG
            
        Returns:
            Dictionary with detailed race information including horses, drivers, odds
        """
        race_details = {
            'race_id': race_id,
            'horses': []
        }
        
        try:
            # Construct race-specific URL
            url = f"https://www.atg.se/spel/{race_id}"
            logger.info(f"Fetching details for race {race_id}")
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse race metadata
            race_details.update(self._parse_race_metadata(soup))
            
            # Parse horse entries
            horse_rows = soup.find_all('tr', class_=re.compile(r'horse|runner|starter'))
            
            for row in horse_rows:
                horse_data = self._parse_horse_entry(row)
                if horse_data:
                    race_details['horses'].append(horse_data)
            
            logger.info(f"Found {len(race_details['horses'])} horses in race {race_id}")
            
        except Exception as e:
            logger.error(f"Error scraping race details for {race_id}: {e}")
        
        self._sleep()
        return race_details
    
    def _parse_race_metadata(self, soup) -> Dict:
        """Parse race-level metadata"""
        metadata = {}
        
        try:
            # These selectors are templates - adjust based on actual HTML
            metadata['distance'] = self._extract_text(soup, re.compile('distance|distans'))
            metadata['prize_money'] = self._extract_text(soup, re.compile('prize|pris'))
            metadata['monte'] = self._extract_text(soup, re.compile('monte|type'))
            metadata['conditions'] = self._extract_text(soup, re.compile('condition|villkor'))
            
        except Exception as e:
            logger.error(f"Error parsing race metadata: {e}")
        
        return metadata
    
    def _parse_horse_entry(self, row_element) -> Optional[Dict]:
        """Parse individual horse entry from race program"""
        try:
            horse_data = {
                'number': self._extract_text(row_element, re.compile('number|startnummer')),
                'horse_name': self._extract_text(row_element, re.compile('horse.*name|häst')),
                'driver': self._extract_text(row_element, re.compile('driver|kusk')),
                'trainer': self._extract_text(row_element, re.compile('trainer|tränare')),
                'odds': self._extract_text(row_element, re.compile('odds|spelvärde')),
                'post_position': self._extract_text(row_element, re.compile('post|spår')),
                'distance': self._extract_text(row_element, re.compile('distance|distans|tillägg')),
                'record': self._extract_text(row_element, re.compile('record|rekord')),
                'starts': self._extract_text(row_element, re.compile('starts|starter')),
                'wins': self._extract_text(row_element, re.compile('wins|segrar')),
            }
            
            return horse_data if horse_data['horse_name'] else None
            
        except Exception as e:
            logger.error(f"Error parsing horse entry: {e}")
            return None
    
    def _extract_text(self, element, pattern) -> Optional[str]:
        """Helper to extract text from element with class pattern"""
        try:
            found = element.find(class_=pattern)
            if found:
                return found.get_text(strip=True)
            # Try finding by text content
            found = element.find(string=pattern)
            if found and found.parent:
                return found.parent.get_text(strip=True)
        except:
            pass
        return None
    
    def scrape_travsport_results(self, date: str) -> List[Dict]:
        """
        Scrape historical results from travsport.se
        
        Args:
            date: Date string in YYYY-MM-DD format
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        try:
            # Travsport results URL structure
            url = f"https://www.travsport.se/sок?date={date}"
            logger.info(f"Fetching Travsport results for {date}")
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse results - adjust selectors as needed
            result_items = soup.find_all('div', class_=re.compile('result|race'))
            
            for item in result_items:
                result_data = self._parse_result_item(item)
                if result_data:
                    results.append(result_data)
            
            logger.info(f"Found {len(results)} results for {date}")
            
        except Exception as e:
            logger.error(f"Error scraping Travsport results for {date}: {e}")
        
        self._sleep()
        return results
    
    def _parse_result_item(self, item_element) -> Optional[Dict]:
        """Parse individual result item"""
        try:
            result = {
                'race_id': self._extract_text(item_element, re.compile('race.*id')),
                'track': self._extract_text(item_element, re.compile('track|bana')),
                'winner': self._extract_text(item_element, re.compile('winner|vinnare')),
                'time': self._extract_text(item_element, re.compile('time|tid')),
                'win_odds': self._extract_text(item_element, re.compile('odds')),
            }
            
            return result if any(result.values()) else None
            
        except Exception as e:
            logger.error(f"Error parsing result item: {e}")
            return None
    
    def save_to_csv(self, data: List[Dict], filename: str):
        """Save scraped data to CSV"""
        if not data:
            logger.warning(f"No data to save to {filename}")
            return
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8')
        logger.info(f"Saved {len(data)} records to {filename}")
    
    def save_to_json(self, data: List[Dict], filename: str):
        """Save scraped data to JSON"""
        if not data:
            logger.warning(f"No data to save to {filename}")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(data)} records to {filename}")


def main():
    """Example usage"""
    
    # Initialize scraper with 2-second delay between requests
    scraper = SwedishTrottingScraper(delay=2.0)
    
    # Get upcoming race dates
    dates = scraper.get_race_days(days_ahead=3)
    
    all_races = []
    
    # Scrape programs for upcoming dates
    for date in dates:
        logger.info(f"Processing date: {date}")
        
        # Get race program
        races = scraper.scrape_atg_program(date)
        all_races.extend(races)
        
        # Optional: Get detailed information for each race
        # for race in races:
        #     if 'race_id' in race:
        #         details = scraper.scrape_race_details(race['race_id'])
        #         # Process details as needed
    
    # Save to CSV
    scraper.save_to_csv(all_races, 'atg_races.csv')
    scraper.save_to_json(all_races, 'atg_races.json')
    
    # Example: Scrape historical results
    # results = scraper.scrape_travsport_results('2025-01-01')
    # scraper.save_to_csv(results, 'travsport_results.csv')


if __name__ == "__main__":
    main()
