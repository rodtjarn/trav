#!/usr/bin/env python3
"""
Check what main race is scheduled for this Saturday
"""

from datetime import datetime, timedelta
from atg_api_scraper import ATGAPIScraper

def get_next_saturday():
    """Get the next Saturday's date"""
    today = datetime.now()
    days_ahead = 5 - today.weekday()  # Saturday is 5
    if days_ahead <= 0:
        days_ahead += 7
    return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

def check_saturday_main_race():
    """Check Saturday's main race"""
    saturday = get_next_saturday()

    print("="*80)
    print(f"SATURDAY MAIN RACE CHECK - {saturday}")
    print("="*80)

    scraper = ATGAPIScraper(delay=0.5)

    # Check for V-games
    game_types = ['V75', 'GS75', 'V86', 'V65']

    print(f"\nChecking ATG calendar for {saturday}...")

    try:
        calendar = scraper.get_calendar_for_date(saturday)

        if not calendar:
            print(f"\n❌ No calendar data available for {saturday}")
            return

        games = calendar.get('games', {})

        print(f"\n{'='*80}")
        print("SCHEDULED V-GAMES")
        print(f"{'='*80}\n")

        found_main_race = False

        # Check for V75 (main Saturday event)
        if 'V75' in games and games['V75']:
            v75 = games['V75'][0]
            race_ids = v75.get('races', [])

            if race_ids:
                print("🏆 V75 - SATURDAY MAIN EVENT")
                print(f"   Game ID: {v75.get('id')}")
                print(f"   Races: {len(race_ids)}")
                print(f"   Start time: {v75.get('startTime', 'TBD')}")

                # Get track info
                tracks = calendar.get('tracks', [])
                for track in tracks:
                    track_races = track.get('races', [])
                    if any(r.get('id') in race_ids for r in track_races):
                        print(f"   Track: {track.get('name')}")
                        break

                found_main_race = True
                print()

        # Check for GS75 (Grand Slam - special event)
        if 'GS75' in games and games['GS75']:
            gs75 = games['GS75'][0]
            race_ids = gs75.get('races', [])

            if race_ids:
                print("🌟 GS75 - GRAND SLAM SPECIAL EVENT")
                print(f"   Game ID: {gs75.get('id')}")
                print(f"   Races: {len(race_ids)}")
                print(f"   Start time: {gs75.get('startTime', 'TBD')}")

                # This is a multi-track event
                print(f"   Multi-track event across Sweden")

                found_main_race = True
                print()

        # Check for V86
        if 'V86' in games and games['V86']:
            v86 = games['V86'][0]
            race_ids = v86.get('races', [])

            if race_ids:
                print("⭐ V86 - MAJOR RACE")
                print(f"   Game ID: {v86.get('id')}")
                print(f"   Races: {len(race_ids)}")
                print(f"   Start time: {v86.get('startTime', 'TBD')}")

                tracks = calendar.get('tracks', [])
                for track in tracks:
                    track_races = track.get('races', [])
                    if any(r.get('id') in race_ids for r in track_races):
                        print(f"   Track: {track.get('name')}")
                        break

                found_main_race = True
                print()

        # Check for V65
        if 'V65' in games and games['V65']:
            v65 = games['V65'][0]
            race_ids = v65.get('races', [])

            if race_ids:
                print("📊 V65 - Standard Race")
                print(f"   Game ID: {v65.get('id')}")
                print(f"   Races: {len(race_ids)}")
                print(f"   Start time: {v65.get('startTime', 'TBD')}")

                tracks = calendar.get('tracks', [])
                for track in tracks:
                    track_races = track.get('races', [])
                    if any(r.get('id') in race_ids for r in track_races):
                        print(f"   Track: {track.get('name')}")
                        break

                print()

        if not found_main_race:
            print("ℹ️  No major V-game (V75/GS75/V86) scheduled")
            print("   Check V65 or use --track option for individual races")

        # Show all tracks racing
        print(f"{'='*80}")
        print("ALL TRACKS RACING")
        print(f"{'='*80}\n")

        tracks = calendar.get('tracks', [])
        for track in tracks:
            track_races = track.get('races', [])
            if track_races:
                print(f"  {track.get('name'):20s} - {len(track_races)} races")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    print(f"\n{'='*80}")
    print("TO BET ON SATURDAY")
    print(f"{'='*80}\n")

    print("# Bet on the main event (if available):")
    print(f"python create_bet.py --game V75")
    print()
    print("# Bet on any V-game:")
    print(f"python create_bet.py --game V65")
    print()
    print("# Bet on specific track:")
    print(f"python create_bet.py --track solvalla")
    print()
    print("# Bet on Saturday's date:")
    print(f"python create_bet.py --date {saturday}")
    print()


if __name__ == '__main__':
    check_saturday_main_race()
