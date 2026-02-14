import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("SpotifyFeatures.csv")

print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())
df.dropna(how="all", inplace=True)

print("Duplicates before:", df.duplicated(subset="track_id").sum())
df.drop_duplicates(subset="track_id", inplace=True)
print("Duplicates after:", df.duplicated(subset="track_id").sum())

df.reset_index(drop=True, inplace=True)
print("Final dataset shape:", df.shape)


# -----------------------------
# Feature Columns
# -----------------------------
audio_features = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence"
]


# -----------------------------
# Mood Labeling Function
# -----------------------------
def assign_mood(row):
    if row["energy"] >= 0.75 and row["tempo"] >= 120:
        return "Energetic"
    elif row["valence"] >= 0.6 and row["energy"] >= 0.5:
        return "Happy"
    elif row["acousticness"] >= 0.6 and row["energy"] <= 0.5:
        return "Calm"
    elif row["valence"] <= 0.4 and row["energy"] <= 0.5:
        return "Sad"
    else:
        return "Calm"


df["mood"] = df.apply(assign_mood, axis=1)

print("\nMood Distribution:")
print(df["mood"].value_counts())


# -----------------------------
# Scaling
# -----------------------------
X = df[audio_features]
y = df["mood"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=audio_features)


# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Logistic Regression
# -----------------------------
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
print("\n========== Logistic Regression ==========")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))


# -----------------------------
# KNN (Best)
# -----------------------------
knn = KNeighborsClassifier(n_neighbors=7, weights="distance", metric="euclidean")
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
print("\n========== KNN ==========")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))


# -----------------------------
# Naive Bayes
# -----------------------------
nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)
print("\n========== Naive Bayes ==========")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_nb))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))


# -----------------------------
# Random Forest (Overfitting analysis)
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("\n========== Random Forest ==========")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


# -----------------------------
# Save Best Model + Scaler
# -----------------------------
joblib.dump(knn, "knn_mood_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model and scaler saved using joblib.")


# -----------------------------
# Visualizations
# -----------------------------
df["mood"].value_counts().plot(kind="bar")
plt.title("Mood Distribution")
plt.xlabel("Mood")
plt.ylabel("Number of Songs")
plt.show()

plt.figure(figsize=(8, 4))
plt.hist(df["energy"], bins=30)
plt.title("Distribution of Energy Feature")
plt.xlabel("Energy")
plt.ylabel("Frequency")
plt.show()

df.boxplot(column="energy", by="mood")
plt.title("Energy vs Mood")
plt.xlabel("Mood")
plt.ylabel("Energy")
plt.show()

cm = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=knn.classes_,
            yticklabels=knn.classes_)
plt.title("Confusion Matrix – KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

models = ["Logistic Regression", "KNN", "Naive Bayes"]
accuracy = [
    accuracy_score(y_test, y_pred_lr),
    accuracy_score(y_test, y_pred_knn),
    accuracy_score(y_test, y_pred_nb)
]
plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()
