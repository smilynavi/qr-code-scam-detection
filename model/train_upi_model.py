import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================
# PATH SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "upi_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "upi_scam_model.pkl"
)


# ============================================================
# SUSPICIOUS KEYWORDS
# ============================================================

SUSPICIOUS_KEYWORDS = [
    "free",
    "gift",
    "winner",
    "win",
    "prize",
    "claim",
    "reward",
    "urgent",
    "verify",
    "verification",
    "password",
    "security",
    "bonus",
    "lottery",
    "cash"
]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(upi_id, merchant, amount):

    upi_id = str(upi_id).strip()
    merchant = str(merchant).strip()
    amount = str(amount).strip()

    # --------------------------------------------------------
    # Combined text
    # --------------------------------------------------------

    combined_text = (
        upi_id.lower() + " " +
        merchant.lower()
    )

    # --------------------------------------------------------
    # 1. UPI ID length
    # --------------------------------------------------------

    upi_id_length = len(upi_id)

    # --------------------------------------------------------
    # 2. Merchant name length
    # --------------------------------------------------------

    merchant_length = len(merchant)

    # --------------------------------------------------------
    # 3. Amount present
    # --------------------------------------------------------

    amount_present = 0

    if amount != "":
        amount_present = 1

    if amount.lower() in [
        "not specified",
        "none",
        "null",
        "nan"
    ]:
        amount_present = 0

    # --------------------------------------------------------
    # 4. Amount numeric
    # --------------------------------------------------------

    try:
        float(amount)
        amount_numeric = 1
    except:
        amount_numeric = 0

    # --------------------------------------------------------
    # 5. Amount greater than 10000
    # --------------------------------------------------------

    amount_high = 0

    try:
        if float(amount) > 10000:
            amount_high = 1
    except:
        amount_high = 0

    # --------------------------------------------------------
    # 6. Valid UPI format
    # --------------------------------------------------------

    import re

    upi_pattern = r"^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$"

    if re.match(upi_pattern, upi_id):
        valid_upi = 1
    else:
        valid_upi = 0

    # --------------------------------------------------------
    # 7. @ count
    # --------------------------------------------------------

    at_count = upi_id.count("@")

    # --------------------------------------------------------
    # 8. Suspicious keyword count
    # --------------------------------------------------------

    suspicious_keyword_count = 0

    for keyword in SUSPICIOUS_KEYWORDS:

        if keyword in combined_text:
            suspicious_keyword_count += 1

    # --------------------------------------------------------
    # 9. Merchant empty
    # --------------------------------------------------------

    if merchant == "":
        merchant_empty = 1
    else:
        merchant_empty = 0

    # --------------------------------------------------------
    # 10. Unusual character count
    # --------------------------------------------------------

    unusual_character_count = 0

    for char in upi_id:

        if not (
            char.isalnum()
            or char in "._-@"
        ):
            unusual_character_count += 1

    # --------------------------------------------------------
    # 11. Merchant suspicious keyword count
    # --------------------------------------------------------

    merchant_suspicious_count = 0

    merchant_lower = merchant.lower()

    for keyword in SUSPICIOUS_KEYWORDS:

        if keyword in merchant_lower:
            merchant_suspicious_count += 1

    # --------------------------------------------------------
    # 12. UPI ID suspicious keyword count
    # --------------------------------------------------------

    upi_suspicious_count = 0

    upi_lower = upi_id.lower()

    for keyword in SUSPICIOUS_KEYWORDS:

        if keyword in upi_lower:
            upi_suspicious_count += 1

    # --------------------------------------------------------
    # 13. Digit count in UPI ID
    # --------------------------------------------------------

    digit_count = 0

    for char in upi_id:

        if char.isdigit():
            digit_count += 1

    # --------------------------------------------------------
    # 14. Digit ratio
    # --------------------------------------------------------

    if len(upi_id) > 0:

        digit_ratio = (
            digit_count /
            len(upi_id)
        )

    else:

        digit_ratio = 0

    # --------------------------------------------------------
    # 15. Payment/request related keyword count
    # --------------------------------------------------------

    payment_keywords = [
        "request",
        "collect",
        "payment",
        "pay"
    ]

    payment_keyword_count = 0

    for keyword in payment_keywords:

        if keyword in combined_text:
            payment_keyword_count += 1

    # --------------------------------------------------------
    # FINAL FEATURE LIST
    # --------------------------------------------------------

    features = [
        upi_id_length,
        merchant_length,
        amount_present,
        amount_numeric,
        amount_high,
        valid_upi,
        at_count,
        suspicious_keyword_count,
        merchant_empty,
        unusual_character_count,
        merchant_suspicious_count,
        upi_suspicious_count,
        digit_count,
        digit_ratio,
        payment_keyword_count
    ]

    return features


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("UPI SCAM DETECTION - MACHINE LEARNING TRAINING")
print("=" * 60)

print()

print("Loading dataset...")

if not os.path.exists(DATASET_PATH):

    print("ERROR: Dataset not found!")
    print()
    print("Expected location:")
    print(DATASET_PATH)

    exit()


df = pd.read_csv(DATASET_PATH)


print()
print("UPI DATASET LOADED")
print("Rows:", len(df))
print("Columns:", list(df.columns))


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "upi_id",
    "merchant",
    "amount",
    "label"
]

for column in required_columns:

    if column not in df.columns:

        print()
        print("ERROR: Missing column:", column)
        print()
        print("Required columns:")
        print(required_columns)

        exit()


# ============================================================
# CLEAN DATA
# ============================================================

df["upi_id"] = (
    df["upi_id"]
    .fillna("")
    .astype(str)
)

df["merchant"] = (
    df["merchant"]
    .fillna("")
    .astype(str)
)

df["amount"] = (
    df["amount"]
    .fillna("")
    .astype(str)
)


# ============================================================
# CONVERT LABEL
# ============================================================

def convert_label(value):

    value = str(value).strip().lower()

    if value in [
        "0",
        "safe",
        "legitimate",
        "normal",
        "good",
        "benign"
    ]:

        return 0

    if value in [
        "1",
        "scam",
        "fraud",
        "phishing",
        "malicious",
        "suspicious",
        "risk"
    ]:

        return 1

    try:
        return int(float(value))

    except:

        return -1


df["label"] = df["label"].apply(convert_label)


# ============================================================
# REMOVE INVALID LABELS
# ============================================================

df = df[
    df["label"].isin([0, 1])
].copy()


# ============================================================
# DISPLAY LABEL DISTRIBUTION
# ============================================================

print()
print("LABEL DISTRIBUTION")
print(
    df["label"].value_counts()
)


# ============================================================
# CREATE FEATURES
# ============================================================

print()
print("Extracting UPI features...")

X = []

for _, row in df.iterrows():

    features = extract_features(
        row["upi_id"],
        row["merchant"],
        row["amount"]
    )

    X.append(features)


X = np.array(X)

y = df["label"].values


print()
print("FEATURE EXTRACTION COMPLETED")
print("Feature Shape:", X.shape)

print()
print("Number of features:", X.shape[1])


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = [

    "upi_id_length",

    "merchant_length",

    "amount_present",

    "amount_numeric",

    "amount_high",

    "valid_upi_format",

    "at_count",

    "suspicious_keyword_count",

    "merchant_empty",

    "unusual_character_count",

    "merchant_suspicious_keyword_count",

    "upi_suspicious_keyword_count",

    "digit_count",

    "digit_ratio",

    "payment_keyword_count"
]


print()
print("FEATURES USED:")
print("-" * 60)

for index, name in enumerate(feature_names):

    print(
        index + 1,
        ".",
        name
    )


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

print()
print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# CREATE RANDOM FOREST MODEL
# ============================================================

print()
print("Creating Random Forest model...")

model = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    class_weight="balanced",

    n_jobs=-1
)


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("Training UPI scam detection model...")

model.fit(
    X_train,
    y_train
)


print()
print("MODEL TRAINING COMPLETED")


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

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


print()
print("=" * 60)
print("UPI MODEL PERFORMANCE")
print("=" * 60)

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


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

importance = model.feature_importances_

feature_importance = list(
    zip(
        feature_names,
        importance
    )
)

feature_importance.sort(
    key=lambda x: x[1],
    reverse=True
)

for name, value in feature_importance:

    print(
        f"{name:<40} : {value:.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

if not os.path.exists(MODEL_DIR):

    os.makedirs(MODEL_DIR)


model_data = {

    "model": model,

    "feature_names": feature_names,

    "suspicious_keywords": SUSPICIOUS_KEYWORDS

}


joblib.dump(
    model_data,
    MODEL_PATH
)


# ============================================================
# SUCCESS MESSAGE
# ============================================================

print()
print("=" * 60)
print("UPI MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print()
print("Model saved at:")

print(MODEL_PATH)

print()
print("Training completed successfully!")

print()
print("=" * 60)