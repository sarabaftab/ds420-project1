# DS 420 Project 1: Phishing URL Classification

Team: Muhammad Sarab Aftab, Saurya Jonchhe, Nipun Yonjan

This project classifies URLs as benign or phishing from lexical and structural URL features. The dataset is `Dataset.csv` (116,600 URLs, 26 columns). The target is `label`: 0 = benign, 1 = phishing.

## Algorithm assignments

Each person investigates two algorithms. Where two people use the same algorithm, the settings must be different.

| Person | Algorithms |
| --- | --- |
| Sarab | Logistic Regression + Decision Tree |
| Saurya | Logistic Regression with different regularization/settings + Linear SVM |
| Nipun | Decision Tree with different depth/splitting settings + Linear SVM with different C/class-weight settings |

After the investigation notebooks, the team picks the top three algorithms. Each person trains one of those with grid search.

## Shared data preparation

Later notebooks should import the same split from `dataset_preparation.py`. Fit the pipeline on the training split only, then transform dev and test.

```python
from dataset_preparation import prepare_datasets

data = prepare_datasets()
pipeline = data["pipeline"]
X_train_prepared = pipeline.fit_transform(data["X_train"])
X_dev_prepared = pipeline.transform(data["X_dev"])
X_test_prepared = pipeline.transform(data["X_test"])
```

## Repository files

- `Dataset.csv` — labeled URL dataset
- `dataset_EDA.ipynb` — exploratory data analysis
- `dataset_preparation.py` — shared cleaning, split, and scaling pipeline
- `project1.pdf` — project requirements
- `TeamReport.pdf` — approved team proposal
