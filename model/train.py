"""Train the WattWise Random Forest on pseudo-labels and save artifacts.
Usage: python model/train.py [data_dir]"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
import utils as U

t0 = time.time()
path = U.find_dataset(sys.argv[1] if len(sys.argv) > 1 else None)
print("Dataset:", path)
raw = U.load_raw(path)
n_rows_file = int(raw["Global_active_power"].notna().sum())
df = U.add_features(raw)
print(f"Grid minutes: {len(df):,}  valid rows: {int(df['valid'].sum()):,}  "
      f"missing/absent minutes: {int(raw['Global_active_power'].isna().sum()):,}")

# Chronological split over valid rows; thresholds fitted on training rows only
valid = df[df["valid"]]
cut = int(len(valid) * U.TRAIN_FRACTION)
split_ts = valid.index[cut]
train_df, test_df = df[df.index < split_ts], df[df.index >= split_ts]
th = U.fit_thresholds(train_df)
print("Thresholds:", th)
df["label"] = U.assign_labels(df, th)
train, test = df.loc[train_df.index].query("valid"), df.loc[test_df.index].query("valid")
print("Train label shares:\n", train["label"].value_counts(normalize=True).round(4))
print("Test label shares:\n", test["label"].value_counts(normalize=True).round(4))

model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
model.fit(train[U.FEATURES].to_numpy("float32"), train["label"])
pred = U.predict(model, test)
acc = accuracy_score(test["label"], pred)
f1 = f1_score(test["label"], pred, average="macro")
cm = confusion_matrix(test["label"], pred, labels=U.CLASSES)
rep = classification_report(test["label"], pred, labels=U.CLASSES, output_dict=True, zero_division=0)
print(classification_report(test["label"], pred, labels=U.CLASSES, zero_division=0))

# Demo days: complete days (>=1400 valid minutes), preferring the test period,
# scored by the smallest class share (so all three classes are visible).
day = df[df["valid"]].copy()
day["pred"] = U.predict(model, day)
g = day.groupby(day.index.date)["pred"].value_counts().unstack(fill_value=0)
for c in U.CLASSES:
    if c not in g: g[c] = 0
g["n"] = g[U.CLASSES].sum(axis=1)
g = g[g["n"] >= 1400]
g["score"] = g[U.CLASSES].min(axis=1)
g["in_test"] = [pd.Timestamp(d) >= split_ts.normalize() for d in g.index]
cand = g[(g[U.CLASSES] >= 30).all(axis=1)]
cand = cand.sort_values(["in_test", "score"], ascending=False)
demo = [{"date": str(d), **{c: int(r[c]) for c in U.CLASSES}, "in_test": bool(r["in_test"])}
        for d, r in cand.head(5).iterrows()]
print("Demo days:", demo[:3])

dates = sorted({str(d) for d in valid.index.date})
meta = {
    "dataset": "UCI Individual Household Electric Power Consumption", "file": path.name,
    "rows_in_file": n_rows_file, "grid_minutes": int(len(df)),
    "valid_rows": int(df["valid"].sum()),
    "missing_minutes": int(raw["Global_active_power"].isna().sum()),
    "first_ts": str(df.index[0]), "last_ts": str(df.index[-1]),
    "train_start": str(train.index[0]), "train_end": str(train.index[-1]),
    "test_start": str(test.index[0]), "test_end": str(test.index[-1]),
    "n_train": int(len(train)), "n_test": int(len(test)),
    "features": U.FEATURES, "classes": U.CLASSES, "thresholds": th,
    "label_params": {"HIGH_Q": U.HIGH_Q, "LOW_Q": U.LOW_Q, "STABLE_Q": U.STABLE_Q,
                     "WINDOW": U.WINDOW},
    "train_label_share": train["label"].value_counts(normalize=True).round(4).to_dict(),
    "test_label_share": test["label"].value_counts(normalize=True).round(4).to_dict(),
    "accuracy": float(acc), "macro_f1": float(f1),
    "confusion_matrix": cm.tolist(), "report": rep,
    "feature_importance": dict(zip(U.FEATURES, map(float, model.feature_importances_))),
    "rf_params": {"n_estimators": 100, "max_depth": 12, "random_state": 42},
    "demo_days": demo, "available_dates": dates,
}
U.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, U.ARTIFACT_DIR / "wattwise_rf.joblib", compress=3)
(U.ARTIFACT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
print(f"Saved artifacts. Accuracy={acc:.4f} MacroF1={f1:.4f}  ({time.time()-t0:.0f}s)")
