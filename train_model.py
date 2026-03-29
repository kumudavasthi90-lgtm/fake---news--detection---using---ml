import pandas as pd
import pickle
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

# Labels
fake["label"] = 0
real["label"] = 1

# Combine datasets
data = pd.concat([fake, real], axis=0)

# Combine title and text
data["content"] = data["title"] + " " + data["text"]

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Cleaning function
def clean_text(text):

    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    return text


data["content"] = data["content"].apply(clean_text)

X = data["content"]
y = data["label"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

Xv_train = vectorizer.fit_transform(X_train)
Xv_test = vectorizer.transform(X_test)

# Model
model = PassiveAggressiveClassifier(max_iter=50)

model.fit(Xv_train, y_train)

# Prediction
pred = model.predict(Xv_test)

print("Accuracy:", accuracy_score(y_test, pred))

# Save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")