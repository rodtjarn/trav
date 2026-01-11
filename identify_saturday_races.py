#!/usr/bin/env python3
"""
Identify main Saturday races (V75, V86) from V-game tagged data
"""

import pandas as pd
from datetime import datetime, timedelta
import sys

def identify_saturday_races(data_file='temporal_processed_data_vgame.csv'):
    """
    Identify main Saturday racing events

    Saturday is the biggest racing day in Sweden:
    - V75: Main event (7 races, huge pools 30-100M SEK)
    - V86: Sometimes on Saturday (8 races, 20-60M SEK)
    - GS75: Special events (4 times per year, 100M+ SEK)
    """
    print("="*80)
    print("IDENTIFYING SATURDAY MAIN RACES")
    print("="*80)

    # Load data
    print(f"\nLoading {data_file}...")
    df = pd.read_csv(data_file)

    if 'vgame_type' not in df.columns:
        print("❌ Error: Data doesn't have V-game tags!")
        print("   Run add_vgame_tags.py first")
        sys.exit(1)

    print(f"Loaded {len(df):,} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Filter for Saturdays (day_of_week = 5)
    saturdays = df[df['day_of_week'] == 5].copy()
    print(f"\nSaturday data: {len(saturdays):,} rows across {saturdays['date'].nunique()} Saturdays")

    # Count V-games on Saturdays
    saturday_vgames = saturdays[saturdays['is_vgame'] == True]
    print(f"Saturday V-game starts: {len(saturday_vgames):,}")

    # Analyze by V-game type
    print("\n" + "="*80)
    print("SATURDAY V-GAME BREAKDOWN")
    print("="*80)

    vgame_counts = saturday_vgames.groupby('vgame_type')['race_id'].nunique()
    for game_type, race_count in sorted(vgame_counts.items()):
        date_count = saturday_vgames[saturday_vgames['vgame_type'] == game_type]['date'].nunique()
        print(f"\n{game_type}:")
        print(f"  {date_count} Saturdays with {game_type}")
        print(f"  {race_count} total races")

    # Find V75 Saturdays (main event)
    print("\n" + "="*80)
    print("V75 SATURDAYS (MAIN EVENT)")
    print("="*80)

    v75_saturdays = saturday_vgames[saturday_vgames['vgame_type'] == 'V75']
    v75_dates = sorted(v75_saturdays['date'].unique())

    print(f"\nFound {len(v75_dates)} V75 Saturdays in 2025")

    if len(v75_dates) > 0:
        print("\nV75 Schedule:")
        for i, date in enumerate(v75_dates[:20], 1):  # Show first 20
            date_df = v75_saturdays[v75_saturdays['date'] == date]
            tracks = date_df['track_name'].value_counts()
            main_track = tracks.index[0] if len(tracks) > 0 else 'Unknown'
            print(f"  {i:2d}. {date} - {main_track}")

        if len(v75_dates) > 20:
            print(f"  ... and {len(v75_dates) - 20} more")

    # Find V86 Saturdays
    print("\n" + "="*80)
    print("V86 SATURDAYS")
    print("="*80)

    v86_saturdays = saturday_vgames[saturday_vgames['vgame_type'] == 'V86']
    v86_dates = sorted(v86_saturdays['date'].unique())

    print(f"\nFound {len(v86_dates)} V86 Saturdays in 2025")

    if len(v86_dates) > 0:
        print("\nV86 Schedule (first 15):")
        for i, date in enumerate(v86_dates[:15], 1):
            date_df = v86_saturdays[v86_saturdays['date'] == date]
            tracks = date_df['track_name'].value_counts()
            main_track = tracks.index[0] if len(tracks) > 0 else 'Unknown'
            print(f"  {i:2d}. {date} - {main_track}")

        if len(v86_dates) > 15:
            print(f"  ... and {len(v86_dates) - 15} more")

    # Find GS75 (Grand Slam)
    print("\n" + "="*80)
    print("GS75 SATURDAYS (GRAND SLAM - SPECIAL EVENTS)")
    print("="*80)

    gs75_saturdays = saturday_vgames[saturday_vgames['vgame_type'] == 'GS75']
    gs75_dates = sorted(gs75_saturdays['date'].unique())

    print(f"\nFound {len(gs75_dates)} GS75 Saturdays in 2025")

    if len(gs75_dates) > 0:
        print("\nGS75 Schedule:")
        for i, date in enumerate(gs75_dates, 1):
            date_df = gs75_saturdays[gs75_saturdays['date'] == date]
            tracks = date_df['track_name'].value_counts()
            main_track = tracks.index[0] if len(tracks) > 0 else 'Unknown'
            print(f"  {i}. {date} - {main_track} (MAJOR EVENT)")

    # Summary for upcoming Saturday
    print("\n" + "="*80)
    print("NEXT SATURDAY ANALYSIS")
    print("="*80)

    today = datetime.now()

    # Find next Saturday
    days_ahead = 5 - today.weekday()  # Saturday is 5
    if days_ahead <= 0:
        days_ahead += 7

    next_saturday = (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
    print(f"\nNext Saturday: {next_saturday}")

    # Check if in dataset
    next_sat_data = saturdays[saturdays['date'] == next_saturday]
    if len(next_sat_data) > 0:
        next_vgames = next_sat_data[next_sat_data['is_vgame'] == True]

        if len(next_vgames) > 0:
            print("\nScheduled V-games:")
            for game_type in next_vgames['vgame_type'].unique():
                if pd.notna(game_type):
                    game_df = next_vgames[next_vgames['vgame_type'] == game_type]
                    tracks = game_df['track_name'].value_counts()
                    main_track = tracks.index[0] if len(tracks) > 0 else 'Unknown'
                    print(f"  - {game_type} at {main_track}")
        else:
            print("\n  No V-games scheduled (regular racing)")
    else:
        print("\n  No data available for this date")

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)

    return {
        'v75_dates': v75_dates,
        'v86_dates': v86_dates,
        'gs75_dates': gs75_dates,
        'next_saturday': next_saturday
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Identify Saturday main races')
    parser.add_argument('--data', default='temporal_processed_data_vgame.csv', help='V-game tagged data file')

    args = parser.parse_args()

    identify_saturday_races(args.data)
