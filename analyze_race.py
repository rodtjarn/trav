#!/usr/bin/env python3
"""
Analyze a specific race and show detailed predictions
"""

import sys
import argparse
from predict_v85 import V85Predictor
from atg_api_scraper import ATGAPIScraper

def main():
    parser = argparse.ArgumentParser(description='Analyze a specific race')
    parser.add_argument('--date', required=True, help='Race date (YYYY-MM-DD)')
    parser.add_argument('--track', required=True, help='Track name')
    parser.add_argument('--race', type=int, help='Race number (optional)')
    args = parser.parse_args()

    # Initialize predictor
    predictor = V85Predictor(
        model_path='temporal_rf_model.pkl',
        metadata_path='temporal_rf_metadata.json'
    )

    print(f"🔍 Analyzing races at {args.track.upper()} on {args.date}")
    print()

    # Get calendar for the date
    scraper = ATGAPIScraper(delay=0.5)
    calendar = scraper.get_calendar_for_date(args.date)

    if not calendar or not calendar.get('tracks'):
        print(f"❌ No races found for {args.date}")
        return

    # Find the track
    track_found = None
    track_name_lower = args.track.lower()
    track_name_normalized = (track_name_lower
        .replace('å', 'a').replace('ä', 'a').replace('ö', 'o'))

    for track in calendar.get('tracks', []):
        track_calendar_name = track.get('name', '').lower()
        track_calendar_normalized = (track_calendar_name
            .replace('å', 'a').replace('ä', 'a').replace('ö', 'o'))

        if track_name_lower in track_calendar_name or track_name_normalized in track_calendar_normalized:
            track_found = track
            break

    if not track_found:
        print(f"❌ Track '{args.track}' not found on {args.date}")
        return

    races = track_found.get('races', [])
    if not races:
        print(f"❌ No races found at {track_found.get('name')}")
        return

    print(f"✅ Found {len(races)} races at {track_found.get('name')}")
    print()

    # Analyze each race (or specific race if provided)
    for i, race in enumerate(races, 1):
        if args.race and i != args.race:
            continue

        race_id = race.get('id')
        print("=" * 100)
        print(f"RACE {i} - {race.get('name', 'Unknown')}")
        print(f"Race ID: {race_id}")
        print(f"Distance: {race.get('distance')}m | Start: {race.get('startMethod', 'Unknown')}")
        print("=" * 100)
        print()

        # Get race details
        race_data = scraper.get_race_details(race_id)
        if not race_data:
            print(f"⚠️  Could not fetch race details")
            continue

        # Process race
        df = predictor.processor.process_race_data([race_data], predictor.feature_cols)
        if df.empty:
            print(f"⚠️  Could not process race data")
            continue

        # Get predictions
        predictions = predictor.model.predict_proba(df[predictor.feature_cols])

        # Display results
        results = []
        for idx, row in df.iterrows():
            prob = predictions[idx][1]  # Probability of winning
            results.append({
                'start_number': int(row['start_number']),
                'horse_name': row['horse_name'],
                'driver': f"{row.get('driver_first_name', '')} {row.get('driver_last_name', '')}".strip(),
                'trainer': f"{row.get('trainer_first_name', '')} {row.get('trainer_last_name', '')}".strip(),
                'post_position': int(row['post_position']) if 'post_position' in row else None,
                'probability': prob,
                'driver_win_rate': row.get('driver_win_rate', 0),
                'record_time': row.get('record_time', 0),
            })

        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)

        # Display
        print(f"{'Rank':<6} {'#':<4} {'Horse':<25} {'Driver':<25} {'Win%':<8} {'Odds Est.':<10}")
        print("-" * 100)

        for rank, r in enumerate(results, 1):
            win_pct = r['probability'] * 100
            # Estimate fair odds from probability
            estimated_odds = max(1.5, min(99.0, 1.0 / r['probability'])) if r['probability'] > 0.01 else 99.0

            marker = "🔥" if rank == 1 else "⭐" if rank <= 3 else "  "
            print(f"{marker} {rank:<4} {r['start_number']:<4} {r['horse_name']:<25} {r['driver'][:24]:<25} {win_pct:>6.1f}% {estimated_odds:>9.1f}")

        print()
        print(f"Model's top pick: #{results[0]['start_number']} {results[0]['horse_name']} ({results[0]['probability']*100:.1f}% win probability)")
        print()

        # Show driver/trainer insights for top 3
        print("📊 Top 3 Insights:")
        print("-" * 100)
        for rank, r in enumerate(results[:3], 1):
            driver_wr = r['driver_win_rate'] * 100 if r['driver_win_rate'] else 0
            record = r['record_time']
            print(f"  {rank}. #{r['start_number']} {r['horse_name']}")
            print(f"     Driver: {r['driver']} (Win rate: {driver_wr:.1f}%)")
            print(f"     Trainer: {r['trainer']}")
            if record > 0:
                print(f"     Record: {int(record//60)}:{int(record%60):02d}.{int((record%1)*10)}")
            print()

        if args.race:
            break  # Only show one race if specific race requested

if __name__ == '__main__':
    main()
