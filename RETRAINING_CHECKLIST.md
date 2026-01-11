# Model Retraining Checklist

Use this checklist when retraining the model to ensure proper train/test separation.

## ✅ Pre-Training Checklist

### 1. Data Collection

- [ ] Data spans at least 60 days (90+ days recommended)
- [ ] Data includes completed races with results (`finish_place` column exists)
- [ ] Data includes V-game tags (if using `collect_vgame_data.py`)
- [ ] No duplicate race_ids in dataset
- [ ] Data file size is reasonable (>10MB for good training)

**Verify:**
```bash
# Check data size
ls -lh temporal_processed_data.csv

# Check date range
head -2 temporal_processed_data.csv | tail -1
tail -1 temporal_processed_data.csv

# Check for results column
head -1 temporal_processed_data.csv | grep -o "finish_place"

# Count unique races
cut -d',' -f1 temporal_processed_data.csv | sort -u | wc -l
```

### 2. Train/Test Split Planning

- [ ] Chosen a split date (e.g., `--train-end 2025-10-31`)
- [ ] Training period is at least 70% of data
- [ ] Test period is at least 14 days
- [ ] No overlap between train and test dates
- [ ] Test period represents FUTURE races (comes after training)

**Example splits:**

| Total Data | Training Period | Test Period | Split Date |
|------------|----------------|-------------|------------|
| 60 days | 48 days | 12 days | 48 days from start |
| 90 days | 70 days | 20 days | 70 days from start |
| 365 days | 300 days | 65 days | Oct 27 (if starting Jan 1) |

### 3. Environment Check

- [ ] Python virtual environment activated
- [ ] Required packages installed (`sklearn`, `pandas`, `imblearn`)
- [ ] Sufficient disk space (at least 1GB free)
- [ ] Model files will be backed up before overwriting

**Verify:**
```bash
# Check virtual environment
which python  # Should show venv path

# Check packages
python -c "import sklearn; import pandas; import imblearn; print('✓ All packages installed')"

# Check disk space
df -h .
```

## ✅ Training Execution

### 4. Run Training Command

```bash
# EXAMPLE - Adjust dates for your data
python train_vgame_model.py \
  --data temporal_processed_data.csv \
  --train-end 2025-10-31 \
  --estimators 100 \
  --depth 20
```

**Command template:**
```bash
python train_vgame_model.py \
  --data <YOUR_DATA_FILE> \
  --train-end <LAST_TRAINING_DATE> \
  [--test-start <FIRST_TEST_DATE>] \
  [--estimators <NUM_TREES>] \
  [--depth <MAX_DEPTH>]
```

### 5. Monitor Training Output

Watch for these key messages:

- [ ] ✓ "Loaded X samples" (should be thousands)
- [ ] ✓ "Selected X features" (should be 50-300)
- [ ] ✓ "Training: YYYY-MM-DD to YYYY-MM-DD"
- [ ] ✓ "Testing: YYYY-MM-DD to YYYY-MM-DD"
- [ ] ✓ "No date overlap - temporal split is valid"
- [ ] ✓ "Training complete!"
- [ ] ✓ "Per-race accuracy: XX.X%"

**Red flags:**

- ❌ "WARNING: Date overlap detected" → Data leakage!
- ❌ Per-race accuracy < 10% → Model not learning
- ❌ Test samples < 1,000 → Need more test data
- ❌ Training samples < 5,000 → Need more training data

## ✅ Post-Training Validation

### 6. Review Metrics

Expected ranges (on test set):

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Per-race accuracy | 22-25% | 18-22% | <18% |
| ROC-AUC | 0.70-0.75 | 0.65-0.70 | <0.65 |
| Races tested | 500+ | 200-500 | <200 |
| High-conf win rate (prob>=0.35) | 35-45% | 28-35% | <28% |

- [ ] Per-race accuracy in acceptable range
- [ ] ROC-AUC > 0.65
- [ ] Tested on at least 200 races
- [ ] Confusion matrix shows reasonable balance

### 7. Check Output Files

- [ ] `vgame_rf_model.pkl` created
- [ ] `vgame_rf_metadata.json` created
- [ ] `vgame_rf_model_test_predictions.csv` created
- [ ] File sizes reasonable (model 1-50MB)

**Verify:**
```bash
ls -lh vgame_rf_model*

# Check metadata
cat vgame_rf_metadata.json | grep -E "per_race_accuracy|roc_auc|test_samples"

# Sample test predictions
head -10 vgame_rf_model_test_predictions.csv
```

### 8. Validate Test Predictions

Check test predictions file for sanity:

```bash
# Count correct predictions
grep ",True" vgame_rf_model_test_predictions.csv | wc -l

# Count total predictions
wc -l vgame_rf_model_test_predictions.csv

# Calculate win rate (manual)
# correct / total should match per_race_accuracy
```

Or use Python:
```python
import pandas as pd

df = pd.read_csv('vgame_rf_model_test_predictions.csv')

print(f"Total test races: {len(df)}")
print(f"Correct predictions: {df['was_correct'].sum()}")
print(f"Win rate: {df['was_correct'].mean()*100:.1f}%")

# Check by confidence level
for threshold in [0.25, 0.30, 0.35]:
    high = df[df['predicted_prob'] >= threshold]
    print(f"\nProb >= {threshold}: {len(high)} races, {high['was_correct'].mean()*100:.1f}% win rate")
```

### 9. Feature Importance Review

- [ ] Top features make sense (not random IDs or names)
- [ ] Mix of different feature types (temporal, stats, position)
- [ ] No single feature dominates (>50% importance)

**Check:**
```bash
# Feature importance is logged during training
# Look for "TOP 25 MOST IMPORTANT FEATURES" in output
```

Expected top features:
- Driver/horse rolling stats
- Post position features
- Recent form indicators
- Track-specific features
- Temporal features (hour, day_of_week)

## ✅ Deployment

### 10. Backup Current Model

```bash
# Backup existing model
cp temporal_rf_model.pkl temporal_rf_model.pkl.backup
cp temporal_rf_metadata.json temporal_rf_metadata.json.backup

# Add timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cp temporal_rf_model.pkl "backups/temporal_rf_model_${TIMESTAMP}.pkl"
```

### 11. Deploy New Model

```bash
# Replace production model
cp vgame_rf_model.pkl temporal_rf_model.pkl
cp vgame_rf_metadata.json temporal_rf_metadata.json

# Verify
ls -lh temporal_rf_model.pkl
```

### 12. Test Betting Tool

- [ ] Betting tool loads new model successfully
- [ ] Predictions complete without errors
- [ ] Probabilities in reasonable range (0.1-0.5)
- [ ] Recommendations make sense

**Test:**
```bash
# Test with V65 (most common)
python create_bet.py --game V65

# Should show:
# - "Loaded model from temporal_rf_model.pkl"
# - Model info with correct date
# - Reasonable horse recommendations
# - Win probabilities 15-45%
```

## ✅ Documentation

### 13. Record Training Details

Create a log entry:

```bash
echo "$(date +%Y-%m-%d): Retrained model" >> MODEL_CHANGELOG.md
echo "  Training: $(grep train_start_date vgame_rf_metadata.json)" >> MODEL_CHANGELOG.md
echo "  Test: $(grep test_start_date vgame_rf_metadata.json)" >> MODEL_CHANGELOG.md
echo "  Per-race accuracy: $(grep per_race_accuracy vgame_rf_metadata.json)" >> MODEL_CHANGELOG.md
echo "" >> MODEL_CHANGELOG.md
```

### 14. Update Documentation

- [ ] Updated model performance metrics in README
- [ ] Noted any significant changes in feature engineering
- [ ] Recorded any issues encountered

## Common Issues & Solutions

### Issue: "No date overlap - temporal split is valid" not showing

**Problem:** Possible data leakage

**Solution:**
```bash
# Check your split dates
python -c "
import pandas as pd
df = pd.read_csv('temporal_processed_data.csv')
print(f'Data spans: {df[\"date\"].min()} to {df[\"date\"].max()}')
"

# Adjust --train-end to be well before data end
```

### Issue: Per-race accuracy < 15%

**Problem:** Model not learning useful patterns

**Solutions:**
1. Check data quality (results present?)
2. Increase training data amount
3. Try different parameters
4. Review feature engineering

### Issue: Model file very large (>100MB)

**Problem:** Too many trees or too deep

**Solutions:**
```bash
# Reduce complexity
python train_vgame_model.py \
  --data temporal_processed_data.csv \
  --train-end 2025-10-31 \
  --estimators 50 \
  --depth 15
```

### Issue: Test predictions file empty or very small

**Problem:** Not enough test data

**Solutions:**
- Adjust `--train-end` to leave more data for testing
- Collect more recent data
- Use `--test-start` to specify test period

## Final Verification

Before using the new model in production:

- [ ] Completed all checklist items above
- [ ] Test set performance is acceptable
- [ ] Old model backed up
- [ ] New model deployed and tested
- [ ] Betting tool works with new model
- [ ] Training logged and documented

**Sign-off:**
```
Model retrained: [DATE]
Trained by: [NAME]
Test accuracy: [XX.X%]
Deployed: [YES/NO]
```

## Quick Reference

**Full retraining command:**
```bash
# Retrain with existing data
python train_vgame_model.py \
  --data temporal_processed_data.csv \
  --train-end 2025-10-31 \
  --estimators 100 \
  --depth 20

# Deploy
cp vgame_rf_model.pkl temporal_rf_model.pkl
cp vgame_rf_metadata.json temporal_rf_metadata.json

# Test
python create_bet.py --game V65
```

**Emergency rollback:**
```bash
# Restore backup
cp temporal_rf_model.pkl.backup temporal_rf_model.pkl
cp temporal_rf_metadata.json.backup temporal_rf_metadata.json
```
