#!/usr/bin/env python3
"""
List available tracks from ATG calendar
"""

from datetime import datetime, timedelta
from atg_api_scraper import ATGAPIScraper

scraper = ATGAPIScraper(delay=0.2)

print("Searching for tracks in the next 7 days...")
print()

all_tracks = set()
today = datetime.now()

for i in range(7):
    date = (today + timedelta(days=i)).strftime('%Y-%m-%d')
    calendar = scraper.get_calendar_for_date(date)

    if calendar and calendar.get('tracks'):
        for track in calendar['tracks']:
            track_name = track.get('name')
            num_races = len(track.get('races', []))
            if num_races > 0:
                all_tracks.add(track_name)
                print(f"{date}: {track_name} ({num_races} races)")

print()
print("=" * 60)
print(f"Unique tracks found: {len(all_tracks)}")
print("=" * 60)
for track in sorted(all_tracks):
    print(f"  {track}")
