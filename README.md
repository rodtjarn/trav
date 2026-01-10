# Swedish Trotting Race Prediction - Data Collection & ML Pipeline

A complete toolkit for scraping, processing, and predicting Swedish trotting race results using Random Forest and other machine learning models.

## ⚠️ Important Legal Notice

**BEFORE YOU START:**
1. ✅ Check `robots.txt` at each website (e.g., `https://www.atg.se/robots.txt`)
2. ✅ Review the Terms of Service for ATG.se and travsport.se
3. ✅ Use rate limiting (included in scripts) to be respectful
4. ✅ For commercial use, contact ATG about their partner API program
5. ✅ This is for educational/research purposes only

## 📁 Project Structure

```
.
├── atg_scraper.py          # Main web scraping script
├── html_inspector.py       # HTML structure analyzer
├── data_processor.py       # Data cleaning & feature engineering
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Inspect Website Structure

Before scraping, understand the HTML structure:

```bash
# Inspect today's race program
python html_inspector.py

# Inspect specific URL
python html_inspector.py "https://www.atg.se/spel/kalender/2025-01-15"
```

This will:
- Show all CSS classes and IDs
- Identify data attributes
- Look for JSON/API endpoints
- Save the full HTML for manual review

### 3. Update Selectors

Based on the inspector output, update the CSS selectors in `atg_scraper.py`:

```python
# Example updates needed in atg_scraper.py
race_cards = soup.find_all('div', class_='actual-race-card-class')
horse_rows = soup.find_all('tr', class_='actual-horse-row-class')
```

### 4. Scrape Data

```bash
# Run the scraper
python atg_scraper.py
```

This will scrape the next 3 days of race programs and save to:
- `atg_races.csv`
- `atg_races.json`

### 5. Process Data for ML

```bash
# Process scraped data
python data_processor.py
```

This creates `processed_trotting_data.csv` with ML-ready features.

## 📊 Data Fields to Collect

### Essential Race Information ("Programmet")

From the race program, collect:

**Race Level:**
- `race_id` - Unique identifier
- `track` - Venue name (Solvalla, Åby, Jägersro, etc.)
- `race_number` - Race number in the program
- `start_time` - Race start time
- `distance` - Race distance (usually 1640m, 2140m, etc.)
- `prize_money` - Prize pool
- `race_type` - Monte/Auto (standing/flying start)
- `track_condition` - Track surface condition
- `temperature` - Weather conditions

**Horse Level (for each starter):**
- `horse_name` - Horse's name
- `number` - Race number (startnummer)
- `post_position` - Starting position (spår)
- `driver` - Driver name (kusk)
- `trainer` - Trainer name (tränare)
- `distance_handicap` - Distance handicap if applicable (tillägg)
- `record` - Personal best time
- `age` - Horse's age
- `sex` - Horse's sex
- `starts` - Total career starts
- `wins` - Total career wins (segrar)
- `earnings` - Career earnings
- `last_5_races` - Recent race history
- `odds` - Current betting odds

### Historical Results

For training data, also scrape:
- `finish_position` - Final placement
- `finish_time` - Final time
- `margin` - Distance behind winner
- `gallop` - Did the horse break gait? (important in trotting!)
- `payout` - Win payout

## 🎯 Key Features for Random Forest Model

The `data_processor.py` creates these important features:

### Temporal Features
- Hour of day
- Day of week
- Weekend indicator
- Month/season

### Post Position Features
- Actual post position
- Inside post indicator (1-5)
- Position ratio (position/total_horses)

### Driver Statistics (90-day rolling)
- Starts count
- Win rate
- Average finish position
- Top-3 rate

### Horse Form (last 5 races)
- Average position
- Best/worst position
- Wins in last 5
- Days since last race
- Average speed

### Speed Metrics
- Personal record speed (km/min)
- Recent average speed
- Speed relative to track

### Track Features
- One-hot encoded track/venue

## 🔧 Advanced Usage

### Scrape Historical Data

Modify `atg_scraper.py` to collect historical data:

```python
# In main() function
from datetime import datetime, timedelta

# Get dates for the last 30 days
start_date = datetime.now() - timedelta(days=30)
dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') 
         for i in range(30)]

all_races = []
for date in dates:
    races = scraper.scrape_atg_program(date)
    all_races.extend(races)
    # Get results from travsport
    results = scraper.scrape_travsport_results(date)
    # Merge races with results
```

### Custom Scraping Schedule

```python
import schedule
import time

def scrape_job():
    scraper = SwedishTrottingScraper()
    # Scrape today's races
    today = datetime.now().strftime('%Y-%m-%d')
    races = scraper.scrape_atg_program(today)
    scraper.save_to_csv(races, f'races_{today}.csv')

# Schedule daily scraping
schedule.every().day.at("06:00").do(scrape_job)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## 📈 Building the ML Model

Once you have processed data, build your Random Forest model:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd

# Load processed data
df = pd.read_csv('processed_trotting_data.csv')

# Separate features and target
feature_cols = [col for col in df.columns 
                if col not in ['is_winner', 'is_top3', 'finish_position']]
X = df[feature_cols]
y = df['is_winner']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply SMOTE for class balance
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_balanced, y_train_balanced)

# Evaluate
from sklearn.metrics import classification_report, roc_auc_score

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
```

## 🔍 Troubleshooting

### Website Structure Changed
If scraping fails:
1. Run `html_inspector.py` to check current structure
2. Update CSS selectors in `atg_scraper.py`
3. Check if ATG has launched a new website version

### Rate Limiting / Blocked
If you get blocked:
1. Increase delay between requests (default is 2 seconds)
2. Use proxy rotation
3. Consider the official ATG API partner program

### Missing Data
Some fields may not always be available:
- Check if the race is too far in the future (odds not set yet)
- Some historical data may be incomplete
- Validate data quality before ML training

## 📚 Useful Resources

- **ATG Official Site:** https://www.atg.se
- **Svensk Travsport:** https://www.travsport.se
- **Travronden (News):** https://www.travronden.se
- **V75 Information:** Special 7-race betting format
- **Elitloppet:** Sweden's premier trotting race

## 🤝 Contributing

Improvements welcome! Key areas:
- Better HTML selectors for current ATG structure
- Additional feature engineering ideas
- Model performance optimizations
- Historical data collection methods

## ⚖️ Disclaimer

This tool is for **educational and research purposes only**. 

- Gambling involves risk
- Past performance does not guarantee future results
- Use responsibly and within your means
- Check local gambling regulations
- Respect website terms of service

## 📝 License

Use responsibly. Not for commercial use without proper agreements with data providers.
