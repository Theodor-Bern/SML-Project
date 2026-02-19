import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score
)
from sklearn.inspection import permutation_importance

df = pd.read_csv('train_data_final.csv')


df['increase_stock'] = df['increase_stock'].map({'high_bike_demand':1, 'low_bike_demand':0})

#Droppa increase_stock-kolumnen från input
X = df.drop('increase_stock', axis=1)
y = df['increase_stock']

numeric_features = ['temp', 'humidity', 'windspeed', 'dew', 'precip', 'snow', 'snowdepth', 'cloudcover', 'visibility']
categorical_features = ['hour_of_day', 'day_of_week', 'month', 'holiday', 'weekday', 'summertime']

preprocessor_sensitive = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

preprocessor_tree = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features), # Fyller NaN med median
        ('cat', SimpleImputer(strategy='constant', fill_value=-1), categorical_features) # Fyller NaN med -1
    ])

# -------------------------
# 1) Train/Test split (test acts like your "outer")
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("=== DATA SPLIT ===")
print(f"Train size: {len(y_train)} | Test size: {len(y_test)}")

def show_balance(name, y_part):
    s = pd.Series(y_part)
    print(f"\nClass balance ({name})")
    vc = s.value_counts().sort_index()
    vcn = s.value_counts(normalize=True).sort_index() #Calculate the normalized amounts of high vs low
    out = pd.DataFrame({"count": vc, "share": vcn})
    out.index = ["LOW(0)", "HIGH(1)"] if list(out.index) == [0, 1] else out.index
    print(out.to_string())

show_balance("TRAIN", y_train)
show_balance("TEST", y_test)

# -------------------------
# 2) Pipeline
# -------------------------
pipe = Pipeline([
    ("preprocessor", preprocessor_sensitive),
    ("classifier", LogisticRegression(max_iter=5000))
])

# -------------------------
# 3) Tuning ONLY on TRAIN using CV (inner CV)
# -------------------------
param_grid = [
    # 1) L2 with lbfgs (strong default baseline)
    {
        "classifier__solver": ["lbfgs"],
        "classifier__penalty": ["l2"],
        "classifier__C": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
        "classifier__class_weight": [None, "balanced"]
    },

    # 2) L1/L2 with liblinear (good for L1, typically fast for smaller problems)
    {
        "classifier__solver": ["liblinear"],
        "classifier__penalty": ["l1", "l2"],
        "classifier__C": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
        "classifier__class_weight": [None, "balanced"]
    },

    # 3) L1/L2 with saga (handles sparse well, flexible)
    {
        "classifier__solver": ["saga"],
        "classifier__penalty": ["l1", "l2"],
        "classifier__C": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
        "classifier__class_weight": [None, "balanced"]
    },

    # 4) ElasticNet with saga (most flexible regularization)
    {
        "classifier__solver": ["saga"],
        "classifier__penalty": ["elasticnet"],
        "classifier__l1_ratio": [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95],
        "classifier__C": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
        "classifier__class_weight": [None, "balanced"]
    }
]

cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="f1_macro",      # F1 for positive class (label=1)
    cv=cv_inner,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\n=== TUNING ON TRAIN (INNER CV) ===")
print("Best parameters:", grid.best_params_)
print(f"Best inner CV macro-F1: {grid.best_score_:.4f}")

best_model = grid.best_estimator_

# Optional: show more CV metrics on TRAIN for the best model (still only train data!)
scoring = {
    "f1_macro": "f1_macro",
    "f1_high":  lambda est, Xv, yv: f1_score(yv, est.predict(Xv), pos_label=1),
    "f1_low":   lambda est, Xv, yv: f1_score(yv, est.predict(Xv), pos_label=0),
    "prec_high":lambda est, Xv, yv: precision_score(yv, est.predict(Xv), pos_label=1, zero_division=0),
    "rec_high": lambda est, Xv, yv: recall_score(yv, est.predict(Xv), pos_label=1, zero_division=0),
}

train_cv = cross_validate(best_model, X_train, y_train, cv=cv_inner, scoring=scoring, n_jobs=-1)

print("\n=== TRAIN CV (best tuned model) ===")
for k in train_cv:
    if k.startswith("test_"):
        vals = train_cv[k]
        print(f"{k.replace('test_','').upper():10s}: mean={vals.mean():.4f} std={vals.std():.4f} | folds={np.round(vals,4)}")

# -------------------------
# 4) Final evaluation on TEST (unseen)
# -------------------------
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
cm_df = pd.DataFrame(
    cm,
    index=["True LOW(0)", "True HIGH(1)"],
    columns=["Pred LOW(0)", "Pred HIGH(1)"]
)

print("\n=== FINAL TEST RESULTS (HOLD-OUT) ===")
print("Confusion matrix (rows=true, cols=pred):")
print(cm_df.to_string())

print("\nClassification report (TEST):")
print(classification_report(
    y_test, y_pred,
    target_names=["LOW(0)", "HIGH(1)"],
    digits=4,
    zero_division=0
))

print("Test summary (explicit):")
print(f"  F1 HIGH(1): {f1_score(y_test, y_pred, pos_label=1):.4f}")
print(f"  F1 LOW(0):  {f1_score(y_test, y_pred, pos_label=0):.4f}")
print(f"  F1 MACRO:   {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"  Precision HIGH(1): {precision_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
print(f"  Recall HIGH(1):    {recall_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")