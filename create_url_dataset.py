import pandas as pd
import os


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

DATASET_PATH = os.path.join(
    DATASET_DIR,
    "qr_dataset.csv"
)


safe_domains = [
    "google.com",
    "microsoft.com",
    "apple.com",
    "amazon.in",
    "wikipedia.org",
    "github.com",
    "stackoverflow.com",
    "linkedin.com",
    "youtube.com",
    "netflix.com",
    "adobe.com",
    "canva.com",
    "python.org",
    "mozilla.org",
    "ubuntu.com",
    "cloudflare.com",
    "dropbox.com",
    "zoom.us",
    "coursera.org",
    "udemy.com",
    "nasa.gov",
    "who.int",
    "un.org",
    "ibm.com",
    "intel.com",
    "oracle.com",
    "samsung.com",
    "sony.com",
    "flipkart.com",
    "irctc.co.in",
    "incometax.gov.in",
    "uidai.gov.in",
    "india.gov.in",
    "sbi.co.in",
    "hdfcbank.com",
    "icicibank.com",
    "axisbank.com",
    "pnbindia.in",
    "bankofbaroda.in",
    "rbi.org.in",
    "sebi.gov.in",
    "licindia.in",
    "irdai.gov.in",
    "digilocker.gov.in",
    "gst.gov.in",
    "tcs.com",
    "infosys.com",
    "wipro.com",
    "zoho.com",
    "freshworks.com"
]


safe_paths = [
    "",
    "/",
    "/home",
    "/about",
    "/contact",
    "/products",
    "/services",
    "/support",
    "/help",
    "/docs",
    "/download",
    "/login",
    "/account",
    "/news",
    "/blog",
    "/careers",
    "/education",
    "/research",
    "/privacy"
]


scam_domains = [
    "freegift.xyz",
    "winnercash.top",
    "claimprize.click",
    "securebank.tk",
    "verifyaccount.xyz",
    "rewardbonus.top",
    "cashwinner.click",
    "prizeclaim.tk",
    "urgentverify.xyz",
    "giftwinner.top"
]


scam_words = [
    "free-gift",
    "winner",
    "claim-prize",
    "urgent",
    "verify-account",
    "secure-bank",
    "reward",
    "cash-bonus",
    "lottery-winner",
    "prize-claim",
    "account-verification",
    "password-reset",
    "bank-security",
    "gift-reward",
    "instant-cash",
    "exclusive-prize",
    "urgent-payment",
    "claim-reward",
    "free-recharge",
    "lucky-winner",
    "bonus-cash",
    "verify-payment",
    "security-alert",
    "account-locked",
    "urgent-verification",
    "free-voucher",
    "winner-cash",
    "claim-gift",
    "prize-winner",
    "reward-claim"
]


rows = []


for i in range(100):

    domain = safe_domains[
        i % len(safe_domains)
    ]

    path = safe_paths[
        i % len(safe_paths)
    ]

    url = (
        "https://www."
        + domain
        + path
    )

    rows.append({
        "url": url,
        "label": 0
    })


for i in range(100):

    domain = scam_domains[
        i % len(scam_domains)
    ]

    word = scam_words[
        i % len(scam_words)
    ]

    url = (
        "http://"
        + domain
        + "/"
        + word
        + "/login?"
        + "account="
        + str(1000 + i)
    )

    rows.append({
        "url": url,
        "label": 1
    })


df = pd.DataFrame(
    rows
)


os.makedirs(
    DATASET_DIR,
    exist_ok=True
)


df.to_csv(
    DATASET_PATH,
    index=False
)


print(
    "URL DATASET CREATED SUCCESSFULLY"
)

print(
    "Dataset path:",
    DATASET_PATH
)

print(
    "Total rows:",
    len(df)
)

print(
    "\nLabel distribution:"
)

print(
    df["label"].value_counts()
)

print(
    "\nColumns:"
)

print(
    list(df.columns)
)