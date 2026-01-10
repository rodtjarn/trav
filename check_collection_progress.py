#!/usr/bin/env python3
"""
Check progress of data collection
"""

import pandas as pd
import os
from datetime import datetime

def check_progress():
    """Check collection progress"""

    # Check for temp file or final file
    files_to_check = [
        'atg_extended_data.csv.temp',
        'atg_extended_data.csv',
        'atg_historical_data.csv'
    ]

    print("="*80)
    print("DATA COLLECTION PROGRESS")
    print("="*80)
    print()

    for filepath in files_to_check:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)

            print(f"📁 {filepath}")
            print(f"   Rows: {len(df):,}")
            print(f"   Races: {df['race_id'].nunique():,}")
            print(f"   Tracks: {df['track_name'].nunique()}")
            print(f"   Date range: {df['race_date'].min()} to {df['race_date'].max()}")
            print(f"   Unique dates: {df['race_date'].nunique()}")

            # Completed races
            completed = df[df['finish_place'].notna()]
            print(f"   Completed starts: {len(completed):,} ({len(completed)/len(df)*100:.1f}%)")

            # Swedish tracks
            swedish = df[df['track_country'] == 'SE']
            print(f"   Swedish tracks: {len(swedish):,} ({len(swedish)/len(df)*100:.1f}%)")

            # Major tracks
            major_tracks = ['Solvalla', 'Jägersro', 'Åby', 'Gävle', 'Eskilstuna',
                          'Bergsåker', 'Färjestad', 'Bollnäs']
            major = df[df['track_name'].isin(major_tracks)]
            print(f"   Major Swedish tracks: {len(major):,} ({len(major)/len(df)*100:.1f}%)")

            # Top tracks
            print(f"\n   Top 10 tracks:")
            for track, count in df['track_name'].value_counts().head(10).items():
                marker = '⭐' if track in major_tracks else ''
                print(f"     {track:20s}: {count:4d} {marker}")

            print()

    # Check if collection is still running
    log_file = '/tmp/claude/-home-per-Work-trav/tasks/ba64234.output'
    if os.path.exists(log_file):
        # Get last few lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            last_lines = lines[-5:] if len(lines) >= 5 else lines

        print("\n📊 RECENT LOG OUTPUT:")
        for line in last_lines:
            print(f"   {line.strip()}")

    print("\n" + "="*80)
    print("To check live progress: tail -f /tmp/claude/-home-per-Work-trav/tasks/ba64234.output")
    print("="*80)

if __name__ == "__main__":
    check_progress()
