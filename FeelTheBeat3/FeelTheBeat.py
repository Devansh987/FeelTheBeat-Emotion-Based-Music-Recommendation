import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv(r"E:\Predictive analysis\FeelTheBeat3\SpotifyFeatures.csv")

print(df.head())
print(df.info())
print(df.describe())


print(df.isnull().sum())
df.dropna(how='all', inplace=True)

print("Duplicates before:", df.duplicated(subset='track_id').sum())
df.drop_duplicates(subset='track_id', inplace=True)
print("Duplicates after:", df.duplicated(subset='track_id').sum())



df.reset_index(drop=True, inplace=True)

print("Final dataset shape:", df.shape)

from sklearn.preprocessing import StandardScaler
audio_features = [
    'acousticness',
    'danceability',
    'energy',
    'instrumentalness',
    'liveness',
    'loudness',
    'speechiness',
    'tempo',
    'valence'
]
X = df[audio_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=audio_features)

def assign_mood(row):
    if row['energy'] >= 0.75 and row['tempo'] >= 120:
        return 'Energetic'
    elif row['valence'] >= 0.6 and row['energy'] >= 0.5:
        return 'Happy'
    elif row['acousticness'] >= 0.6 and row['energy'] <= 0.5:
        return 'Calm'
    elif row['valence'] <= 0.4 and row['energy'] <= 0.5:
        return 'Sad'
    else:
        return 'Calm'

df['mood'] = df.apply(assign_mood, axis=1)

print("\nMood Distribution:")
print(df['mood'].value_counts())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    df['mood'],
    test_size=0.2,
    random_state=42,
    stratify=df['mood']
)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print("Accuracy:", lr_accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_lr))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_lr))


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7,weights='distance',metric='euclidean')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print("Accuracy:", knn_accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_knn))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_knn))

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nNaive Bayes Classification Report:\n",
      classification_report(y_test, y_pred_nb))
print("\nNaive Bayes Confusion Matrix:\n",
      confusion_matrix(y_test, y_pred_nb))




from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nRandom Forest Classification Report:\n",classification_report(y_test, y_pred_rf))
print("\nRandom Forest Confusion Matrix:\n",confusion_matrix(y_test, y_pred_rf))

joblib.dump(knn, "knn_mood_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved using joblib.")



df['mood'].value_counts().plot(kind='bar')
plt.title("Mood Distribution")
plt.xlabel("Mood")
plt.ylabel("Number of Songs")
plt.show()

plt.figure(figsize=(8,4))
plt.hist(df['energy'], bins=30)
plt.title("Distribution of Energy Feature")
plt.xlabel("Energy")
plt.ylabel("Frequency")
plt.show()

df.boxplot(column='energy', by='mood')
plt.title("Energy vs Mood")
plt.xlabel("Mood")
plt.ylabel("Energy")
plt.show()


cm = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm, annot=True, fmt='d',xticklabels=knn.classes_,yticklabels=knn.classes_)
plt.title("Confusion Matrix – KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


models = ['Logistic Regression', 'KNN','Naive Bayes']
accuracy = [accuracy_score(y_test, y_pred_lr),accuracy_score(y_test, y_pred_knn),accuracy_score(y_test, y_pred_nb)]
plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

song_db: pd.DataFrame = df.copy()
song_db[audio_features] = X_scaled
def recommend_songs_by_mood(user_mood, song_db, audio_features, top_n=5):
    mood_songs = song_db[song_db['mood'] == user_mood]

    if len(mood_songs) == 0:
        return "No songs found."

    features = mood_songs[audio_features].values

    ref_idx = np.random.randint(len(mood_songs))
    ref_vector = features[ref_idx].reshape(1, -1)

    similarity = cosine_similarity(ref_vector, features)[0]
    top_indices = similarity.argsort()[::-1][1:top_n+1]

    return mood_songs.iloc[top_indices][
        ['track_name', 'artist_name', 'mood']
    ]

user_mood = 'Sad'
recommendations = recommend_songs_by_mood(
    user_mood,
    song_db,
    audio_features,
    top_n=5
)
print(recommendations)
