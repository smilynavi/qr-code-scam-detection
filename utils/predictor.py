import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def extract_features(url):

    url = str(url).strip().lower()

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


feature_names = [
    "url_length",
    "dot_count",
    "slash_count",
    "hyphen_count",
    "at_count",
    "question_count",
    "equal_count",
    "ampersand_count",
    "percent_count",
    "https",
    "http",
    "login",
    "verify",
    "account",
    "bank",
    "secure",
    "free",
    "winner",
    "claim",
    "password",
    "urgent",
    "reward",
    "gift",
    "prize",
    "bitly",
    "tinyurl",
    "googl"
]


df = pd.read_csv(
    DATASET_PATH
)


df = df.dropna()


if "url" not in df.columns:
    raise ValueError(
        "Dataset must contain 'url' column"
    )


if "label" not in df.columns:
    raise ValueError(
        "Dataset must contain 'label' column"
    )


X = []

for url in df["url"]:

    X.append(
        extract_features(url)
    )


X = np.array(X)

y = df["label"].values


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
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


y_pred = model.predict(
    X_test
)


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
    "Accuracy:",
    accuracy
)

print(
    "Precision:",
    precision
)

print(
    "Recall:",
    recall
)

print(
    "F1 Score:",
    f1
)


if not os.path.exists(
    MODEL_DIR
):

    os.makedirs(
        MODEL_DIR
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
    "URL Random Forest model saved successfully."
)

print(
    "Model path:",
    MODEL_PATH
)