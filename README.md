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

## Shared data preparation

Later notebooks should import the same split from `dataset_preparation.py`. Fit the pipeline on the training split only, then transform dev and test. Investigation notebooks score the dev split only. Leave the test split untouched until the conclusion.

```python
from dataset_preparation import prepare_datasets

data = prepare_datasets()
pipeline = data["pipeline"]
X_train_prepared = pipeline.fit_transform(data["X_train"])
X_dev_prepared = pipeline.transform(data["X_dev"])
X_test_prepared = pipeline.transform(data["X_test"])
```

## Investigation notebooks

The shared preparation file is on `main`. Each person imports `prepare_datasets()`, fits each assigned model on the training split, and scores it on the dev split. Sarab's two algorithms are in separate notebooks.

The data is about 86% benign, so accuracy by itself can hide a weak phishing detector. Report accuracy, plus precision, recall, and F1 for the phishing class, and ROC AUC. End the notebook with a short comparison of which model looks stronger and why.

| Person | Notebook | Algorithms |
| --- | --- | --- |
| Sarab | `Aftab_Sarab_logreg_investigation.ipynb` and `Aftab_Sarab_tree_investigation.ipynb` | Logistic regression and decision tree |
| Saurya | `Jonchhe_Saurya_logreg_investigation.ipynb` and `Jonchhe_Saurya_svm_investigation.ipynb` | Logistic regression with different regularization/settings, and linear SVM |
| Nipun | `Yonjan_Nipun_tree_svm_investigation.ipynb` | Decision tree with different depth/splitting settings, and linear SVM with different C/class-weight settings |

Use the constructors below as written. `random_state=42` stays the same for everyone so the runs can be compared. The shared preparation pipeline also stays the same. These fixed settings are for the investigation notebooks. Grid search comes later.

### Sarab: logistic regression and decision tree

Sarab uses L2 logistic regression with `C=1.0`, the `lbfgs` solver, and no class weighting. The decision tree uses the Gini criterion, grows until the leaves are pure, and looks at every feature at each split.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

log_reg = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    class_weight=None,
    max_iter=1000,
    random_state=42,
)

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    class_weight=None,
    random_state=42,
)
```

### Saurya: logistic regression and linear SVM

Saurya’s logistic regression uses L1 regularization, stronger shrinkage (`C=0.1`), the `liblinear` solver, and `class_weight="balanced"`. `liblinear` is the solver that fits this L1 setup.

Saurya’s linear SVM is `LinearSVC` with `C=1.0` and no class weighting. Use `decision_function` for ROC AUC. `LinearSVC` does not provide `predict_proba`.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

log_reg = LogisticRegression(
    penalty="l1",
    C=0.1,
    solver="liblinear",
    class_weight="balanced",
    max_iter=2000,
    random_state=42,
)

linear_svm = LinearSVC(
    penalty="l2",
    loss="squared_hinge",
    dual=False,
    C=1.0,
    class_weight=None,
    max_iter=5000,
    random_state=42,
)
```

### Nipun: decision tree and linear SVM

Nipun’s decision tree uses entropy, stops at depth 8, needs 50 samples to make a split and 20 samples in each leaf, considers `sqrt` of the features at each split, and uses balanced class weights.

Nipun’s linear SVM is also `LinearSVC` with `penalty="l2"` and `dual=False`. The two values that differ from Saurya’s SVM are `C=0.01` and `class_weight="balanced"`. Use `decision_function` for ROC AUC.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

decision_tree = DecisionTreeClassifier(
    criterion="entropy",
    splitter="best",
    max_depth=8,
    min_samples_split=50,
    min_samples_leaf=20,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
)

linear_svm = LinearSVC(
    penalty="l2",
    loss="squared_hinge",
    dual=False,
    C=0.01,
    class_weight="balanced",
    max_iter=5000,
    random_state=42,
)
```

### Settings that must stay different

| Setting | Sarab logistic regression | Saurya logistic regression |
| --- | --- | --- |
| `penalty` | `"l2"` | `"l1"` |
| `C` | `1.0` | `0.1` |
| `solver` | `"lbfgs"` | `"liblinear"` |
| `class_weight` | `None` | `"balanced"` |
| `max_iter` | `1000` | `2000` |

| Setting | Sarab decision tree | Nipun decision tree |
| --- | --- | --- |
| `criterion` | `"gini"` | `"entropy"` |
| `max_depth` | `None` | `8` |
| `min_samples_split` | `2` | `50` |
| `min_samples_leaf` | `1` | `20` |
| `max_features` | `None` | `"sqrt"` |
| `class_weight` | `None` | `"balanced"` |

| Setting | Saurya linear SVM | Nipun linear SVM |
| --- | --- | --- |
| `C` | `1.0` | `0.01` |
| `class_weight` | `None` | `"balanced"` |
| `penalty` | `"l2"` | `"l2"` |
| `dual` | `False` | `False` |
| `max_iter` | `5000` | `5000` |

After these notebooks, the team picks the top three algorithms. Each person then trains one of those with grid search.

## Repository files

- `Dataset.csv` — labeled URL dataset
- `dataset_EDA.ipynb` — exploratory data analysis
- `dataset_preparation.py` — shared cleaning, split, and scaling pipeline
- `project1.pdf` — project requirements
- `TeamReport.pdf` — approved team proposal
