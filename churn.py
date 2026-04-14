import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score

print("\n" + "="*60)
print("   TELCO CUSTOMER CHURN PREDICTION - ML PROJECT")
print("="*60)

# STEP 1: LOAD DATA
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(f"\n✅ Data Loaded! Rows: {df.shape[0]} | Columns: {df.shape[1]}")

# STEP 2: CLEAN DATA
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print(f"✅ Data Cleaned!")
print(f"   No Churn: {(df['Churn']==0).sum()} | Churned: {(df['Churn']==1).sum()}")

# STEP 3: FEATURE ENGINEERING — create 4 new useful columns
df["ChargesPerMonth"]  = df["TotalCharges"] / (df["tenure"] + 1)
df["HighSpender"]      = (df["MonthlyCharges"] > 70).astype(int)
df["LongTermCustomer"] = (df["tenure"] > 24).astype(int)
df["NewCustomer"]      = (df["tenure"] < 6).astype(int)
print("✅ Feature Engineering Done! Added 4 new features.")

# STEP 4: ENCODE TEXT COLUMNS TO NUMBERS
df = pd.get_dummies(df, drop_first=True)
print(f"✅ Encoding Done! Total features: {df.shape[1] - 1}")

# STEP 5: SPLIT DATA
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Split Done! Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# STEP 6: SCALE FEATURES
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("✅ Scaling Done!")

# STEP 7: TRAIN MODELS
print("\n⏳ Training models...")
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, random_state=42),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42),
}

best_score = 0
best_model = None
best_name  = ""
results    = {}

print(f"\n{'Model':<25} {'Accuracy':>10} {'Churn Recall':>14}")
print("-" * 52)
for name, m in models.items():
    m.fit(X_train_scaled, y_train)
    preds        = m.predict(X_test_scaled)
    acc          = round(accuracy_score(y_test, preds) * 100, 2)
    churn_recall = round(recall_score(y_test, preds) * 100, 2)
    results[name] = {"accuracy": acc, "recall": churn_recall}
    print(f"{name:<25} {acc:>9}% {churn_recall:>13}%")
    if acc > best_score:
        best_score = acc
        best_model = m
        best_name  = name

print(f"\n🏆 Best: {best_name} with {best_score}%")

# STEP 8: HYPERPARAMETER TUNING
print("\n⏳ Tuning best model... (2-3 mins)")
param_grid = {
    "n_estimators"     : [100, 200, 300],
    "max_depth"        : [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
}
grid_search = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    param_grid, cv=5, scoring="recall", n_jobs=-1, verbose=0
)
grid_search.fit(X_train_scaled, y_train)
tuned_model = grid_search.best_estimator_
tuned_preds = tuned_model.predict(X_test_scaled)
tuned_acc   = round(accuracy_score(y_test, tuned_preds) * 100, 2)
print(f"✅ Tuned Accuracy: {tuned_acc}%")
print(f"   Best params: {grid_search.best_params_}")

final_model = tuned_model

# STEP 9: EVALUATION
final_preds = final_model.predict(X_test_scaled)
final_acc   = round(accuracy_score(y_test, final_preds) * 100, 2)
print(f"\n🎯 FINAL ACCURACY: {final_acc}%")
print(classification_report(y_test, final_preds, target_names=["No Churn", "Churned"]))

# STEP 10: CONFUSION MATRIX
cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted: Stay", "Predicted: Churn"],
            yticklabels=["Actual: Stay", "Actual: Churn"])
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("✅ Saved: confusion_matrix.png")

# STEP 11: FEATURE IMPORTANCE
feat_importance = pd.Series(final_model.feature_importances_, index=X.columns)
feat_importance.nlargest(10).plot(kind="barh", figsize=(10, 6), color="steelblue")
plt.title("Top 10 Features That Predict Churn")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("✅ Saved: feature_importance.png")

# STEP 12: SAVE MODEL, SCALER, AND COLUMN ORDER
joblib.dump(final_model, "churn_model.pkl")
print("✅ Model saved: churn_model.pkl")

joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler saved: scaler.pkl")

# ⭐ THIS IS THE KEY FIX — save exact column order for app.py
joblib.dump(list(X.columns), "feature_columns.pkl")
print("✅ Column order saved: feature_columns.pkl")
print(f"   Total columns saved: {len(list(X.columns))}")

print("\n" + "="*60)
print("   ✅ PROJECT COMPLETE!")
print(f"   Final Accuracy : {final_acc}%")
print(f"   Features Used  : {X_train.shape[1]}")
print("="*60)