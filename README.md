# Late Deliveries

QuickBite wants to know, when an order is placed, whether it will arrive late, so it can offer a $4 voucher in advance. This model predicts the probability that an order will be late and flags it if the probability is above a threshold chosen from the company's costs.

## Setup

```bash
conda env create -f environment.yml
conda activate lab05-late-deliveries
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
python -m latedeliveries.train
```

This recreates `models/` and `reports/` from nothing.

## Check it

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Decisions
### Columns

The model is used when an order is placed, so a column is used only if its value is known at that moment.

| Used | How |
|---|---|
| `distance_km`, `items`, `hour`, `courier_months`, `prep_minutes` | numeric: missing values filled with the training median, then scaled |
| `weather`, `day_type`, `vehicle` | category: one-hot encoded, unseen values ignored |

| Removed | Reason |
|---|---|
| `customer_rating` | Given after delivery, so it does not exist when the order is placed. It is also partly caused by lateness itself. |
| `order_id` | Only a label. It says nothing about the order. |
| `late` | The target. |

**Leakage check.** With `customer_rating` added back, LogisticRegression's CV ROC-AUC rises from 0.806 to 0.916. A jump that large from one column is a warning, not good news, so it stays out.

Assumptions about prediction time:
- `prep_minutes` is how long the restaurant actually took to prepare the food. At order time only the restaurant's or dispatcher's estimate exists. Missing values are filled with the training median.
- `vehicle` and `courier_months` assume the courier is assigned when the order is placed.

### Model

Five-fold stratified cross-validation on the training data only (2,250 orders), every model inside the same pipeline. Mean ± standard deviation over the folds (`reports/cv_results.csv`):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| DummyClassifier | 0.844 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| LogisticRegression | 0.860 ± 0.010 | 0.638 ± 0.093 | 0.239 ± 0.034 | 0.348 ± 0.049 | 0.806 ± 0.033 |
| RandomForest | 0.861 ± 0.007 | 0.678 ± 0.089 | 0.211 ± 0.024 | 0.321 ± 0.033 | 0.777 ± 0.028 |
| BalancedDecisionTree | 0.789 ± 0.016 | 0.325 ± 0.045 | 0.327 ± 0.066 | 0.325 ± 0.050 | 0.601 ± 0.031 |

The baseline reaches 0.844 accuracy by never flagging anyone, so accuracy cannot judge this problem. Missing a late order costs $15 and a voucher costs $4, so what matters is how well a model separates late orders from on-time ones; the threshold is chosen later from costs. That is ROC-AUC, and LogisticRegression is highest (0.806). The balanced
tree has the best recall, but its ROC-AUC (0.601) shows it ranks orders poorly.

### Tuning

`GridSearchCV` over `model__C` in {0.1, 1, 10} and `model__class_weight` in {None, "balanced"} (6 combinations), using the same five stratified folds on the training data only, scored on **recall**, because a missed late order costs almost four times a voucher.

Best settings: `C=0.1`, `class_weight="balanced"`. CV recall rose from 0.239 ± 0.034 (untuned) to 0.715, a gain of 0.476, far larger than the fold-to-fold
spread. Most of this gain comes from `class_weight="balanced"`, which pushes predicted
probabilities up, rather than from `C`.

### Threshold

Every flagged order costs $4, late or not, and every late order not flagged costs $15. The default threshold of 0.5 ignores these costs, so we chose our own. Using out-of-fold probabilities on the training rows (`cross_val_predict`), we computed the total cost at every threshold from 0.05 to 0.95 in steps of 0.05 and kept the cheapest: **0.65** (training cost $4,063, against $5,265 for flagging nobody). The test set played no part.

## Result
The final pipeline was fitted on all 2,250 training orders and scored once on the 750
test orders at threshold 0.65 (`reports/metrics.json`).

| | Predicted on time | Predicted late |
|---|---|---|
| **Actually on time** | 570 | 63 |
| **Actually late** | 62 | 55 |

| Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|
| 0.466 | 0.470 | 0.468 | 0.766 |

| Policy | Total cost on the test set |
|---|---|
| Flag nobody (current policy) | $1,755 |
| Flag everybody | $3,000 |
| **Our model** | **$1,402** |

The model saves $353 (20%) against flagging nobody, close to the 23% saving seen
out-of-fold on the training rows. With only 117 late orders in the test set, we trust
that the model is cheaper than flagging nobody, but the exact size of the saving could
easily move by a hundred dollars or more on another sample.

## What the tests cover
`python -m pytest` runs 19 tests.

- **Features** (`test_pipeline.py`): the feature lists are exactly the ones above, and `customer_rating`, `order_id` and `late` are never among them.
- **Split**: 2,250 training and 750 test orders with no overlap, the same rows on every run, and the same late rate in both parts (parametrized over train and test).
- **Pipeline**: an order with a missing `prep_minutes` and `distance_km` and an unseen weather value (`"hail"`) still gets valid probabilities between 0 and 1 that sum to 1.
- **Cost and threshold** (`test_evaluation.py`): small examples worked out by hand, including each of the four cases of late/on time and flagged/not flagged, and both extremes: a case where flagging everybody is cheapest and one where flagging nobody is.
- **Saved model** (`test_saved_model.py`): a pipeline saved with joblib to `tmp_path` predicts exactly the same after reloading.

**Not tested: the full `train.py` run.** It writes to `models/` and `reports/`, and its numbers depend on the data rather than on a rule we can check by hand.  
Instead we check it by running `python -m latedeliveries.train` on a fresh clone and confirming that `reports/metrics.json` matches the committed file.

## Known limitations
- **Invented data, small test set.**  
The data is generated by `data.py`, and the test set holds only 117 late orders, so the size of the saving is uncertain.
- **`prep_minutes` at prediction time.**  
The model learned from how long the restaurant actually took, but that is only known after the food is ready; when an order is placed only an estimate exists. In real use we would also need to predict the restaurant's prep time from past orders, and the model would likely do a little worse with that estimate.
- **Probabilities are not real chances.**   
`class_weight="balanced"` pushes predicted probabilities up, which is why the threshold is 0.65. With more time we would calibrate them so that a user could be shown a real chance of lateness.
- **`hour` is treated as a number.**  
Lateness rises at the lunch and dinner rush and falls again (as is common for restaurants), which a single linear slope on `hour` cannot represent. Treating it as a category, or adding a rush-hour flag, might help.
- **Fixed cost rule.**  
The rule assumes a voucher always works and every missed late order costs exactly $15. If either changes, the threshold must be chosen again.
- **Code layout.** `pipeline.py` defines the features, the split and the preprocessing pipeline. `pipeline_modified.py` holds the model comparison (`compare_models`,  `save_cv_results`) and imports those definitions from `pipeline.py`. Earlier it kept its own copy of the feature list, so one run trained the model without `prep_minutes`; the decision to include it was made before any test result was seen, and the fix was not chosen by comparing test scores. With more time we would rename `pipeline_modified.py` to `compare.py`.
