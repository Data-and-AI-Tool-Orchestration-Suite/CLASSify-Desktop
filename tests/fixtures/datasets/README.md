# Golden Datasets for ML Regression Testing

These CSV files are used by the ML regression test suite to verify that
the training engine produces correct artifacts and sane metrics.

## Datasets

| File | Type | Rows | Features | Classes | Notes |
|---|---|---|---|---|---|
| `binary_classification.csv` | Supervised binary | 30 | 3 | 2 | Linearly separable, numeric features |
| `multiclass.csv` | Supervised multiclass | 45 | 4 | 3 | Three balanced classes |
| `clustering.csv` | Unsupervised | 30 | 3 | N/A | 3 natural clusters for KMeans |
| `missing_values.csv` | Supervised with NaN | 25 | 3 | 2 | Tests drop/constant/synthetic imputation |
| `categorical.csv` | Supervised with categorical | 20 | 3 | 2 | Has yes/no string column (auto-detected as bool) |

## Expected properties

- `binary_classification.csv`: RF accuracy > 0.5 (better than random)
- `multiclass.csv`: All models produce non-NaN metrics
- `clustering.csv`: KMeans produces ≥2 clusters, silhouette > -1
- `missing_values.csv`: Dataset can be loaded; dropping NaNs leaves ≥10 rows
- `categorical.csv`: yes/no column detected as bool by get_column_types_internal

## Usage in tests

```python
@pytest.mark.ml_regression
def test_train_on_binary(golden_csv_path):
    df = pd.read_csv(golden_csv_path / "binary_classification.csv")
    # ... train and assert
```
