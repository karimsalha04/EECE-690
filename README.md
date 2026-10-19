# EECE-690
# baseline code using the phase quantities 
# sweeping to find best L2 regularization coeffecient
# plotting training vs. validation loss to choose the right number of iterations and prevent overfitting 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings for the iteration plot because we are stopping to calculate loss before convergence
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
    mean_absolute_error,
    log_loss,
)

# ============================================================
# 1. CONFIG
# ============================================================
DATA_PATH = r"C:\Users\Karim Salha\Desktop\Masters\Fall 2026\EECE 690\MLdataset\fault_dataset.xlsx"
RANDOM_STATE = 42

# ============================================================
# 2. LOAD DATA
# ============================================================
df = pd.read_excel(DATA_PATH, keep_default_na=False)
df["fault_type"] = df["fault_type"].astype(str)

# ============================================================
# 3. GROUPED SPLIT BY Case_ID  (75 / 10 / 15)
# ============================================================
case_df = (
    df.groupby("Case_ID", as_index=False)
      .agg(case_label=("fault_type", lambda s: next((x for x in s if x != "None"), "None")))
)

train_cases, temp_cases = train_test_split(
    case_df, test_size=0.25, stratify=case_df["case_label"], random_state=RANDOM_STATE
)
val_cases, test_cases = train_test_split(
    temp_cases, test_size=0.60, stratify=temp_cases["case_label"], random_state=RANDOM_STATE
)

train_df = df[df["Case_ID"].isin(train_cases["Case_ID"])].copy()
val_df   = df[df["Case_ID"].isin(val_cases["Case_ID"])].copy()
test_df  = df[df["Case_ID"].isin(test_cases["Case_ID"])].copy() # UNTOUCHED

# ============================================================
# 4. FEATURES (Phase Raw quantities)
# ============================================================
line_features = ["Length_km", "R1_pu", "X1_pu", "R0_pu", "X0_pu"]

phasor_features = []
for q in ["V", "I"]:
    for side in ["send", "recv"]:
        for ph in ["a", "b", "c"]:
            phasor_features.append(f"{q}_{side}_{ph}_mag")
            phasor_features.append(f"{q}_{side}_{ph}_ang")

features = phasor_features + line_features

X_train_cls = train_df[features]
y_train_cls = train_df["fault_type"]
X_val_cls = val_df[features]
y_val_cls = val_df["fault_type"]

# ============================================================
# 5. PLOT LOSS VS ITERATIONS (Convergence Plot)
# ============================================================
print("\n" + "=" * 70)
print("GENERATING CONVERGENCE PLOT (CROSS-ENTROPY LOSS VS ITERATIONS)")
print("=" * 70)

iterations_to_test = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
train_losses = []
val_losses = []

for max_iter in iterations_to_test:
    temp_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            solver="lbfgs", C=1.0, max_iter=max_iter, 
            class_weight="balanced", random_state=RANDOM_STATE
        ))
    ])
    temp_clf.fit(X_train_cls, y_train_cls)
    
    train_prob = temp_clf.predict_proba(X_train_cls)
    val_prob = temp_clf.predict_proba(X_val_cls)
    classes = temp_clf.classes_ 
    
    train_losses.append(log_loss(y_train_cls, train_prob, labels=classes))
    val_losses.append(log_loss(y_val_cls, val_prob, labels=classes))

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(iterations_to_test, train_losses, marker='o', label='Training Loss')
plt.plot(iterations_to_test, val_losses, marker='s', label='Validation Loss')
plt.title('Phase (Raw) Features: Loss vs Iterations')
plt.xlabel('Number of Iterations (max_iter)')
plt.ylabel('Cross-Entropy Loss')
plt.xscale('log') 
plt.xticks(iterations_to_test, labels=iterations_to_test)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 6. TRAIN FINAL MULTINOMIAL LOGISTIC REGRESSION
# ============================================================
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        solver="lbfgs", penalty="l2", C=1.0, max_iter=5000, 
        class_weight="balanced", random_state=RANDOM_STATE
    ))
])
clf.fit(X_train_cls, y_train_cls)
y_val_pred = clf.predict(X_val_cls)

class_order = ["LG", "LL", "LLG", "LLL", "LLLG", "None"]
labels = [c for c in class_order if c in df["fault_type"].unique()]

print("\n" + "=" * 70)
print("VALIDATION CONFUSION MATRIX")
print("=" * 70)
cm_val = confusion_matrix(y_val_cls, y_val_pred, labels=labels)
print(pd.DataFrame(cm_val, index=labels, columns=labels))

print("\nValidation classification report:")
print(classification_report(y_val_cls, y_val_pred, labels=labels, zero_division=0))

# ============================================================
# 7. LINEAR REGRESSION FOR LOCATION (ACTUAL FAULTS)
# ============================================================
train_fault = train_df[train_df["is_faulted"] == 1].copy()
val_fault   = val_df[val_df["is_faulted"] == 1].copy()

X_train_reg = train_fault[features]
y_train_reg = train_fault["location_percent"]
X_val_reg = val_fault[features]
y_val_reg = val_fault["location_percent"]

reg = Pipeline([("scaler", StandardScaler()), ("linreg", LinearRegression())])
reg.fit(X_train_reg, y_train_reg)

y_val_reg_pred = np.clip(reg.predict(X_val_reg), 0.0, 100.0)
abs_err_val = np.abs(y_val_reg_pred - y_val_reg.values)

print("\n" + "=" * 70)
print("VALIDATION LOCATION ERROR — ACTUAL FAULTED ROWS ONLY")
print("=" * 70)
print(f"Mean absolute error: {abs_err_val.mean():.3f} %")
print(f"Max absolute error:  {abs_err_val.max():.3f} %")

# ============================================================
# 8. END-TO-END VALIDATION (classifier -> regression)
# ============================================================
val_pred_df = val_df.copy()
val_pred_df["pred_fault_type"] = clf.predict(val_pred_df[features])
val_pred_df["pred_location_percent"] = np.nan

fault_mask = val_pred_df["pred_fault_type"] != "None"

if fault_mask.any():
    preds = reg.predict(val_pred_df.loc[fault_mask, features])
    val_pred_df.loc[fault_mask, "pred_location_percent"] = np.clip(preds, 0.0, 100.0)

detected_actual = (
    (val_pred_df["is_faulted"] == 1) &
    (val_pred_df["pred_fault_type"] != "None")
)

print("\n" + "=" * 70)
print("END-TO-END VALIDATION LOCATION — DETECTED ACTUAL FAULTS")
print("=" * 70)

if detected_actual.any():
    abs_err_e2e = np.abs(
        val_pred_df.loc[detected_actual, "pred_location_percent"]
        - val_pred_df.loc[detected_actual, "location_percent"]
    )
    print(f"Detected faults: {detected_actual.sum()} / {val_pred_df['is_faulted'].sum()}")
    print(f"Mean absolute error: {abs_err_e2e.mean():.3f} %")
    print(f"Max absolute error:  {abs_err_e2e.max():.3f} %")
else:
    print("No actual faults were detected as faults.")

# ============================================================
# 9. C TUNING FOR LOGISTIC REGRESSION
# ============================================================
print("\n" + "=" * 70)
print("C TUNING FOR LOGISTIC REGRESSION (VALIDATION MACRO F1)")
print("=" * 70)

for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    clf.set_params(logreg__C=C)
    clf.fit(X_train_cls, y_train_cls)
    y_val_pred_c = clf.predict(X_val_cls)
    macro_f1 = f1_score(y_val_cls, y_val_pred_c, labels=labels, average="macro", zero_division=0)
    print(f"C = {C:>6}: validation macro F1 = {macro_f1:.4f}")
