"""
Step 8 - Transition Drivers
Dynamic Customer Segmentation Using Temporal Purchasing Behavior

This script contains the core workflow completed so far:
1. Load the clustered customer profiles.
2. Build consecutive-window transition pairs.
3. Prepare behavioral features and next-period segment labels.
4. Split data chronologically.
5. Train and evaluate a Random Forest classifier.
6. Examine feature importance.
7. Calculate and visualize SHAP values.

Note:
- This is an interim backup of the work completed so far.
- The dataset CSV must be available at the path specified below.
- SHAP plots are predictive explanations, not evidence of causal effects.
"""

# ============================================================
# 1. Import libraries and load data
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATA_PATH = "/content/dynamic_profiles_step6_CLUSTERED_LABELED (1).csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# ============================================================
# 2. Create consecutive-window transition pairs
# ============================================================

df = df.sort_values(["Customer ID", "Window_Num"]).copy()

df["Next_Segment"] = (
    df.groupby("Customer ID")["Segment"].shift(-1)
)
df["Next_Window_Num"] = (
    df.groupby("Customer ID")["Window_Num"].shift(-1)
)

transition_df = df[
    df["Next_Window_Num"] == df["Window_Num"] + 1
].copy()

transition_df = transition_df.dropna(subset=["Next_Segment"])

print("Transition dataset shape:", transition_df.shape)


# ============================================================
# 3. Define behavioral features and target
# ============================================================

features = [
    "Recency",
    "Frequency",
    "Monetary",
    "Product_Diversity",
    "Purchase_Interval",
    "Spending_Trend"
]

target = "Next_Segment"

modeling_df = transition_df[features + [target]].copy()

print("\nModeling dataset shape:", modeling_df.shape)
print("\nTarget distribution:")
print(modeling_df[target].value_counts())


# ============================================================
# 4. Check data quality
# ============================================================

print("\nMissing values:")
print(modeling_df[features].isnull().sum())

print("\nInfinite values:")
print(
    np.isinf(
        modeling_df[features].select_dtypes(include="number")
    ).sum()
)


# ============================================================
# 5. Chronological train-test split
# ============================================================

# Windows 1-16 are used for training.
# Windows 17-21 are reserved for testing.

train_df = transition_df[
    transition_df["Window_Num"] <= 16
].copy()

test_df = transition_df[
    transition_df["Window_Num"] >= 17
].copy()

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

print("\nTraining set:", X_train.shape)
print("Testing set:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ============================================================
# 6. Train the Random Forest model
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("\nRandom Forest training completed.")


# ============================================================
# 7. Predict the next customer segment
# ============================================================

y_pred = rf_model.predict(X_test)

print("Prediction completed.")
print("Number of predictions:", len(y_pred))

sample_predictions = pd.DataFrame({
    "Actual_Segment": y_test.values[:10],
    "Predicted_Segment": y_pred[:10]
})

print("\nSample predictions:")
print(sample_predictions)


# ============================================================
# 8. Evaluate the model
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

labels = rf_model.classes_

conf_matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=labels,
    columns=labels
)

print("\nConfusion Matrix:")
print(conf_matrix_df)


# ============================================================
# 9. Random Forest feature importance
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature importance:")
print(feature_importance)

plt.figure(figsize=(9, 5))
plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)
plt.xlabel("Importance")
plt.ylabel("Behavioral Features")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# ============================================================
# 10. SHAP analysis
# ============================================================

# Use a sample of the test set to reduce computation time.
X_shap = X_test.sample(
    n=min(500, len(X_test)),
    random_state=42
)

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_shap)

print("\nSHAP analysis completed.")
print("SHAP sample size:", X_shap.shape[0])
print("SHAP output shape:", shap_values.shape)


# ============================================================
# 11. SHAP summary plots for each segment
# ============================================================

segment_names = rf_model.classes_

for i, segment in enumerate(segment_names):
    print(f"\nSHAP Summary Plot - {segment}")

    shap.summary_plot(
        shap_values[:, :, i],
        X_shap,
        show=True
    )


# ============================================================
# End of current Step 8 backup
# ============================================================


# ============================================================
# 12. Export SHAP importance tables
# ============================================================

# Calculate mean absolute SHAP values for each segment.
segment_names = rf_model.classes_

shap_importance_by_segment = pd.DataFrame(
    {
        segment: np.abs(shap_values[:, :, i]).mean(axis=0)
        for i, segment in enumerate(segment_names)
    },
    index=features
)

# Convert the values into percentages within each segment.
shap_importance_pct = (
    shap_importance_by_segment
    .div(shap_importance_by_segment.sum(axis=0), axis=1)
    * 100
)

print("\\nSHAP Importance by Segment:")
print(shap_importance_by_segment.round(4))

print("\\nSHAP Importance Percentage by Segment:")
print(shap_importance_pct.round(2))

# Save both tables for reporting.
shap_importance_by_segment.to_csv(
    "shap_importance_by_segment.csv",
    index=True
)

shap_importance_pct.to_csv(
    "shap_importance_percentage_by_segment.csv",
    index=True
)

print("\\nSHAP importance tables exported.")
