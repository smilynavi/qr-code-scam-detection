import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==============================
# PATH CONFIGURATION
# ==============================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "qr_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "scam_model.pkl"
)


# ==============================
# URL FEATURE EXTRACTION
# ==============================

def extract_features(url):

    url = str(url).lower()

    features = [
        len(url),
        url.count("."),
        url.count("/"),
        url.count("-"),
        url.count("@"),
        url.count("?"),
        url.count("="),
        url.count("&"),
        url.count("%"),

        int(url.startswith("https")),
        int(url.startswith("http")),

        int("login" in url),
        int("verify" in url),
        int("account" in url),
        int("bank" in url),
        int("secure" in url),
        int("free" in url),
        int("winner" in url),
        int("claim" in url),
        int("password" in url),
        int("urgent" in url),
        int("reward" in url),
        int("gift" in url),
        int("prize" in url),

        int("bit.ly" in url),
        int("tinyurl" in url),
        int("goo.gl" in url)
    ]

    return features


# ==============================
# LOAD DATASET
# ==============================

print("\nURL DATASET LOADING...")

df = pd.read_csv(
    DATASET_PATH
)

print(
    "Dataset rows:",
    len(df)
)

print(
    "Dataset columns:",
    list(df.columns)
)


# ==============================
# CREATE FEATURES
# ==============================

X = np.array([
    extract_features(url)
    for url in df["url"]
])

y = df["label"].astype(int)


print(
    "\nFEATURE SHAPE:",
    X.shape
)

print(
    "LABEL DISTRIBUTION:"
)

print(
    y.value_counts()
)


# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ==============================
# RANDOM FOREST MODEL
# ==============================

print(
    "\nTRAINING RANDOM FOREST MODEL..."
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# ==============================
# MODEL PREDICTION
# ==============================

y_pred = model.predict(
    X_test
)


# ==============================
# MODEL PERFORMANCE
# ==============================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print(
    "\nURL MODEL PERFORMANCE"
)

print(
    "Accuracy  :",
    round(accuracy, 4)
)

print(
    "Precision :",
    round(precision, 4)
)

print(
    "Recall    :",
    round(recall, 4)
)

print(
    "F1 Score  :",
    round(f1, 4)
)


# ==============================
# FEATURE NAMES
# ==============================

feature_names = [
    "url_length",
    "dot_count",
    "slash_count",
    "dash_count",
    "at_count",
    "question_count",
    "equal_count",
    "ampersand_count",
    "percent_count",
    "https",
    "http",
    "login_keyword",
    "verify_keyword",
    "account_keyword",
    "bank_keyword",
    "secure_keyword",
    "free_keyword",
    "winner_keyword",
    "claim_keyword",
    "password_keyword",
    "urgent_keyword",
    "reward_keyword",
    "gift_keyword",
    "prize_keyword",
    "bitly",
    "tinyurl",
    "googl"
]


# ==============================
# FEATURE IMPORTANCE
# ==============================

print(
    "\nFEATURE IMPORTANCE"
)

importance = model.feature_importances_

for name, value in zip(
    feature_names,
    importance
):
    print(
        name,
        ":",
        round(value, 4)
    )


# ==============================
# SAVE MODEL
# ==============================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model_data = {
    "model": model,
    "feature_names": feature_names
}

joblib.dump(
    model_data,
    MODEL_PATH
)


print(
    "\nURL RANDOM FOREST MODEL SAVED SUCCESSFULLY"
)

print(
    "Model path:",
    MODEL_PATH
)

print(
    "Feature count:",
    len(feature_names)
)