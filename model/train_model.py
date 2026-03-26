import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("data/system_logs.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# =========================
# 2. DATA CLEANING
# =========================
print("\nChecking null values:")
print(df.isnull().sum())

# Remove invalid values
df = df[df["cpu"] <= 100]
df = df[df["memory"] <= 100]
df = df[df["response_time"] > 0]

# =========================
# 3. DATA ANALYSIS
# =========================
print("\nStatistical Summary:")
print(df.describe())

print("\nFailure Distribution:")
print(df["failure"].value_counts())

# Plot response time distribution
df["response_time"].hist()
plt.title("Response Time Distribution")
plt.xlabel("Response Time (ms)")
plt.ylabel("Frequency")
plt.show()

# =========================
# 4. FEATURE ENGINEERING
# =========================
df["error_flag"] = df["status_code"].apply(lambda x: 0 if x == 200 else 1)
df["load"] = (df["cpu"] + df["memory"]) / 2

print("\nAfter Feature Engineering:")
print(df.head())

# =========================
# 5. TRAIN-TEST SPLIT
# =========================
X = df.drop("failure", axis=1)
y = df["failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6. MODEL TRAINING
# =========================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================
# 7. MODEL EVALUATION
# =========================
y_pred = model.predict(X_test)

print("\nModel Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# 8. FEATURE IMPORTANCE
# =========================
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importance:")
print(feature_importance.sort_values(ascending=False))

# =========================
# 9. SAVE MODEL
# =========================
pickle.dump(model, open("model/model.pkl", "wb"))

print("\nModel saved successfully ✅")