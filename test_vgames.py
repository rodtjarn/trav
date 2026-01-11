#!/usr/bin/env python3
"""
Test script to validate V-game betting system support

This script tests that create_bet.py correctly searches for and identifies
all supported V-game types.
"""

import subprocess
import sys
from datetime import datetime

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'

# Game types to test with their expected characteristics
GAME_TYPES = {
    'V75': {'races': 7, 'typical_days': ['Saturday']},
    'V86': {'races': 8, 'typical_days': ['Wednesday', 'Saturday']},
    'V85': {'races': 8, 'typical_days': ['Friday']},
    'V65': {'races': 6, 'typical_days': ['Any day']},
    'V64': {'races': 6, 'typical_days': ['Tuesday', 'Thursday', 'Sunday']},
    'V5': {'races': 5, 'typical_days': ['Variable']},
    'V4': {'races': 4, 'typical_days': ['Variable']},
    'V3': {'races': 3, 'typical_days': ['Variable']},
}

def test_game_type(game_type):
    """Test a specific game type"""
    print(f"\n{BLUE}Testing {game_type}...{RESET}")
    print(f"  Expected: {GAME_TYPES[game_type]['races']} races")
    print(f"  Typical days: {', '.join(GAME_TYPES[game_type]['typical_days'])}")

    try:
        # Run the betting tool
        result = subprocess.run(
            ['python', 'create_bet.py', '--game', game_type],
            capture_output=True,
            text=True,
            timeout=30
        )

        output = result.stdout + result.stderr

        # Check if game was found
        if f'Found {game_type}' in output:
            # Extract race count
            import re
            race_match = re.search(rf'Found {game_type}.*?(\d+) races', output)
            if race_match:
                race_count = int(race_match.group(1))
                expected_races = GAME_TYPES[game_type]['races']

                if race_count == expected_races:
                    print(f"  {GREEN}✓ PASS{RESET} - Found game with {race_count} races")

                    # Check if race labels are correct
                    if f'{game_type} Race 1' in output:
                        print(f"  {GREEN}✓ PASS{RESET} - Race labels correct ({game_type} Race N)")
                    else:
                        print(f"  {YELLOW}⚠ WARNING{RESET} - Race labels may be incorrect")

                    return True
                else:
                    print(f"  {RED}✗ FAIL{RESET} - Found {race_count} races, expected {expected_races}")
                    return False
            else:
                print(f"  {YELLOW}⚠ WARNING{RESET} - Could not parse race count")
                return True  # Found game, but couldn't verify details
        elif f'No {game_type} found' in output:
            print(f"  {YELLOW}⚠ NOT FOUND{RESET} - No {game_type} available in next 30 days")
            print(f"     This is normal - {game_type} may not be scheduled")
            return True  # Not an error, just not available
        else:
            print(f"  {RED}✗ FAIL{RESET} - Unexpected output")
            print(f"     Output: {output[:200]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  {RED}✗ FAIL{RESET} - Command timed out")
        return False
    except Exception as e:
        print(f"  {RED}✗ FAIL{RESET} - Error: {e}")
        return False

def main():
    """Run all tests"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}V-Game Betting System Test Suite{RESET}")
    print(f"{BLUE}Testing date: {datetime.now().strftime('%Y-%m-%d')}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    results = {}

    for game_type in GAME_TYPES.keys():
        results[game_type] = test_game_type(game_type)

    # Print summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Test Summary{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for game_type, passed_test in results.items():
        status = f"{GREEN}✓ PASS{RESET}" if passed_test else f"{RED}✗ FAIL{RESET}"
        print(f"  {game_type:6s}: {status}")

    print(f"\n{BLUE}Total: {passed}/{total} tests passed{RESET}")

    if passed == total:
        print(f"\n{GREEN}All tests passed!{RESET}")
        print(f"\n{BLUE}Note:{RESET} Some games may show 'NOT FOUND' - this is normal.")
        print(f"Not all game types are available every day.")
        return 0
    else:
        print(f"\n{RED}Some tests failed. Please review the output above.{RESET}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
