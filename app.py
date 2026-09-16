from flask import Flask, render_template, request, jsonify
import os
import re
import time
import base64
import requests
import cv2
import numpy as np
import joblib

from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs, unquote


app = Flask(__name__)


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

for filename in [".env", "api.env"]:

    env_path = os.path.join(
        BASE_DIR,
        filename
    )

    if os.path.exists(env_path):

        load_dotenv(
            env_path,
            override=True
        )


# =========================================================
# VIRUSTOTAL API KEY
# =========================================================

VT_API_KEY = os.getenv(
    "VIRUSTOTAL_API_KEY"
)

if VT_API_KEY:

    VT_API_KEY = (
        VT_API_KEY
        .strip()
        .strip('"')
        .strip("'")
    )


print("=" * 55)
print("VIRUSTOTAL API KEY CHECK")
print(
    "API KEY LOADED:",
    bool(VT_API_KEY)
)
print(
    "API KEY LENGTH:",
    len(VT_API_KEY)
    if VT_API_KEY
    else 0
)
print("=" * 55)


# =========================================================
# LOAD MACHINE LEARNING MODEL
# =========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "scam_model.pkl"
)

try:

    ml_model = joblib.load(
        MODEL_PATH
    )

    ML_MODEL_AVAILABLE = True

    print("=" * 55)
    print("MACHINE LEARNING MODEL")
    print("MODEL LOADED: True")
    print(
        "MODEL PATH:",
        MODEL_PATH
    )
    print("=" * 55)

except Exception as e:

    ml_model = None

    ML_MODEL_AVAILABLE = False

    print("=" * 55)
    print("MACHINE LEARNING MODEL")
    print("MODEL LOADED: False")
    print(
        "MODEL ERROR:",
        e
    )
    print("=" * 55)


# =========================================================
# PAGE ROUTES
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route("/qr-scanner")
def qr_scanner():

    return render_template(
        "qr_code.html"
    )


@app.route("/upi-payment")
def upi_payment():

    return render_template(
        "upi_payment.html"
    )


# =========================================================
# URL NORMALIZATION
# =========================================================

def normalize_url(url):

    if not url:

        return None

    url = str(url).strip()

    # Remove spaces
    url = re.sub(
        r"\s+",
        "",
        url
    )

    if not url:

        return None

    # Add HTTPS if protocol is missing
    if not re.match(
        r"^https?://",
        url,
        re.IGNORECASE
    ):

        url = "https://" + url

    try:

        parsed = urlparse(
            url
        )

        if parsed.scheme.lower() not in [
            "http",
            "https"
        ]:

            return None

        if not parsed.netloc:

            return None

        return url

    except Exception:

        return None


# =========================================================
# ML FEATURE EXTRACTION
# MUST MATCH train_model.py
# =========================================================

def extract_ml_features(url):

    url = str(
        url
    ).lower().strip()

    features = [

        # URL structure
        len(url),

        url.count("."),

        url.count("/"),

        url.count("-"),

        url.count("@"),

        url.count("?"),

        url.count("="),

        url.count("&"),

        url.count("%"),

        # Protocol
        1 if url.startswith(
            "https://"
        ) else 0,

        1 if url.startswith(
            "http://"
        ) else 0,

        # Suspicious keywords
        1 if "login" in url else 0,

        1 if "verify" in url else 0,

        1 if "account" in url else 0,

        1 if "bank" in url else 0,

        1 if "secure" in url else 0,

        1 if "free" in url else 0,

        1 if "winner" in url else 0,

        1 if "claim" in url else 0,

        1 if "password" in url else 0,

        1 if "urgent" in url else 0,

        1 if "reward" in url else 0,

        1 if "gift" in url else 0,

        1 if "prize" in url else 0,

        # URL shorteners
        1 if "bit.ly" in url else 0,

        1 if "tinyurl" in url else 0,

        1 if "goo.gl" in url else 0

    ]

    return np.array([
        features
    ])


# =========================================================
# MACHINE LEARNING PREDICTION
# =========================================================

def predict_url_with_ml(url):

    if not ML_MODEL_AVAILABLE:

        return {

            "status": "UNKNOWN",

            "message":
                "Machine Learning model is not available.",

            "prediction": None,

            "scam_probability": 0

        }

    try:

        features = extract_ml_features(
            url
        )

        # Prediction
        prediction = ml_model.predict(
            features
        )[0]

        # Probability
        probability = ml_model.predict_proba(
            features
        )[0]

        scam_probability = (
            probability[1] * 100
        )

        if int(prediction) == 1:

            status = "RISK"

            message = (
                "Machine Learning model "
                "detected this URL as potentially "
                "phishing or malicious."
            )

        else:

            status = "SAFE"

            message = (
                "Machine Learning model "
                "classified this URL as potentially safe."
            )

        return {

            "status": status,

            "message": message,

            "prediction": int(
                prediction
            ),

            "scam_probability":
                round(
                    scam_probability,
                    2
                )

        }

    except Exception as e:

        print(
            "ML PREDICTION ERROR:",
            e
        )

        return {

            "status": "UNKNOWN",

            "message":
                "Machine Learning prediction failed.",

            "prediction": None,

            "scam_probability": 0

        }


# =========================================================
# VIRUSTOTAL URL ID
# =========================================================

def get_vt_url_id(url):

    encoded = base64.urlsafe_b64encode(
        url.encode()
    ).decode()

    return encoded.rstrip("=")


# =========================================================
# RESULT BUILDER
# =========================================================

def make_result(
    status="UNKNOWN",
    message="Unable to determine URL status.",
    malicious=0,
    suspicious=0
):

    return {

        "status":
            status or "UNKNOWN",

        "message":
            message or "",

        "malicious":
            int(
                malicious or 0
            ),

        "suspicious":
            int(
                suspicious or 0
            )

    }


# =========================================================
# LOCAL URL HEURISTIC
# =========================================================

def local_url_check(url):

    reasons = []

    try:

        parsed = urlparse(
            url
        )

        host = parsed.netloc.lower()

        path = parsed.path.lower()

        query = parsed.query.lower()

        full_url = url.lower()

        suspicious_words = [

            "login",
            "verify",
            "verification",
            "account",
            "password",
            "credential",
            "urgent",
            "confirm",
            "security-check",
            "update-account",
            "free-gift",
            "winner",
            "prize",
            "claim",
            "reward",
            "otp"

        ]

        for word in suspicious_words:

            if word in full_url:

                reasons.append(
                    "Suspicious keyword detected: "
                    + word
                )

        # IP address
        ip_pattern = (
            r"^(?:\d{1,3}\.){3}\d{1,3}"
        )

        if re.match(
            ip_pattern,
            host.split(":")[0]
        ):

            reasons.append(
                "Website uses an IP address "
                "instead of a domain name."
            )

        # Too many subdomains
        hostname = host.split(":")[0]

        if hostname.count(".") >= 4:

            reasons.append(
                "Unusually high number of "
                "subdomains detected."
            )

        # @ symbol
        if "@" in url:

            reasons.append(
                "URL contains an @ symbol "
                "which can hide the real destination."
            )

        # Long URL
        if len(url) > 180:

            reasons.append(
                "URL is unusually long."
            )

        return reasons

    except Exception:

        return []


# =========================================================
# BUILD VIRUSTOTAL RESULT
# =========================================================

def result_from_stats(stats):

    malicious = int(
        stats.get(
            "malicious",
            0
        ) or 0
    )

    suspicious = int(
        stats.get(
            "suspicious",
            0
        ) or 0
    )

    if malicious > 0:

        return make_result(

            "RISK",

            (
                f"{malicious} malicious "
                "detection(s) found. "
                "Avoid opening this website."
            ),

            malicious,

            suspicious

        )

    if suspicious > 0:

        return make_result(

            "SUSPICIOUS",

            (
                f"{suspicious} suspicious "
                "detection(s) found. "
                "Verify the website before continuing."
            ),

            malicious,

            suspicious

        )

    return make_result(

        "SAFE",

        (
            "No malicious or suspicious "
            "detections were found by VirusTotal."
        ),

        malicious,

        suspicious

    )


# =========================================================
# VIRUSTOTAL URL SCANNER
# =========================================================

def scan_url_with_virustotal(url):

    print("=" * 55)
    print("VIRUSTOTAL URL SCAN")
    print("URL:", url)
    print(
        "API KEY AVAILABLE:",
        bool(VT_API_KEY)
    )
    print("=" * 55)

    if not VT_API_KEY:

        return make_result(

            "UNKNOWN",

            "VirusTotal API key is missing."

        )

    headers = {

        "x-apikey":
            VT_API_KEY,

        "accept":
            "application/json"

    }

    try:

        # =================================================
        # STEP 1 - CHECK EXISTING URL
        # =================================================

        url_id = get_vt_url_id(
            url
        )

        lookup_url = (
            "https://www.virustotal.com/api/v3/urls/"
            + url_id
        )

        lookup_response = requests.get(

            lookup_url,

            headers=headers,

            timeout=20

        )

        print(
            "URL LOOKUP STATUS:",
            lookup_response.status_code
        )

        if lookup_response.status_code == 200:

            lookup_json = (
                lookup_response.json()
            )

            attributes = (
                lookup_json
                .get("data", {})
                .get("attributes", {})
            )

            stats = attributes.get(
                "last_analysis_stats"
            )

            if stats:

                print(
                    "EXISTING VT RESULT FOUND"
                )

                print(
                    "STATS:",
                    stats
                )

                result = result_from_stats(
                    stats
                )

                # Local heuristic
                if (

                    result["malicious"] == 0

                    and

                    result["suspicious"] == 0

                ):

                    reasons = local_url_check(
                        url
                    )

                    if reasons:

                        result["status"] = (
                            "SUSPICIOUS"
                        )

                        result["message"] = (
                            "VirusTotal found no "
                            "detections, but suspicious "
                            "URL patterns were detected."
                        )

                        result["reasons"] = reasons

                return result

        elif lookup_response.status_code == 401:

            return make_result(

                "UNKNOWN",

                (
                    "VirusTotal API key is "
                    "invalid or unauthorized."
                )

            )

        # =================================================
        # STEP 2 - SUBMIT NEW URL
        # =================================================

        print(
            "Submitting URL to VirusTotal..."
        )

        submit_response = requests.post(

            "https://www.virustotal.com/api/v3/urls",

            headers=headers,

            data={
                "url": url
            },

            timeout=30

        )

        print(
            "SUBMIT STATUS:",
            submit_response.status_code
        )

        if submit_response.status_code not in [
            200,
            201
        ]:

            print(
                "VT SUBMIT RESPONSE:",
                submit_response.text
            )

            if submit_response.status_code == 401:

                message = (
                    "VirusTotal API key is "
                    "invalid or unauthorized."
                )

            elif submit_response.status_code == 429:

                message = (
                    "VirusTotal API request "
                    "limit reached."
                )

            else:

                message = (
                    "VirusTotal API error: "
                    + str(
                        submit_response.status_code
                    )
                )

            return make_result(
                "UNKNOWN",
                message
            )

        submit_json = (
            submit_response.json()
        )

        analysis_id = (
            submit_json
            .get("data", {})
            .get("id")
        )

        if not analysis_id:

            return make_result(

                "UNKNOWN",

                (
                    "VirusTotal did not "
                    "return an analysis ID."
                )

            )

        print(
            "ANALYSIS ID:",
            analysis_id
        )

        # =================================================
        # STEP 3 - WAIT FOR ANALYSIS
        # =================================================

        analysis_url = (

            "https://www.virustotal.com/api/v3/"
            "analyses/"
            + analysis_id

        )

        for attempt in range(20):

            time.sleep(2)

            print(
                "Checking analysis:",
                attempt + 1
            )

            analysis_response = requests.get(

                analysis_url,

                headers=headers,

                timeout=20

            )

            print(
                "ANALYSIS RESPONSE:",
                analysis_response.status_code
            )

            if analysis_response.status_code != 200:

                continue

            analysis_json = (
                analysis_response.json()
            )

            attributes = (
                analysis_json
                .get("data", {})
                .get("attributes", {})
            )

            analysis_status = (
                attributes.get(
                    "status",
                    ""
                )
            )

            print(
                "ANALYSIS STATUS:",
                analysis_status
            )

            if analysis_status != "completed":

                continue

            stats = attributes.get(
                "stats",
                {}
            )

            print(
                "FINAL STATS:",
                stats
            )

            result = result_from_stats(
                stats
            )

            # Local heuristic
            if (

                result["malicious"] == 0

                and

                result["suspicious"] == 0

            ):

                reasons = local_url_check(
                    url
                )

                if reasons:

                    result["status"] = (
                        "SUSPICIOUS"
                    )

                    result["message"] = (
                        "VirusTotal found no "
                        "detections, but suspicious "
                        "URL patterns were detected."
                    )

                    result["reasons"] = reasons

            return result

        # =================================================
        # ANALYSIS STILL PROCESSING
        # =================================================

        print(
            "VirusTotal analysis is "
            "still processing."
        )

        return make_result(

            "UNKNOWN",

            (
                "VirusTotal analysis is still "
                "processing. Please try again "
                "after a few seconds."
            )

        )

    # =====================================================
    # EXCEPTIONS
    # =====================================================

    except requests.exceptions.Timeout:

        return make_result(

            "UNKNOWN",

            "VirusTotal request timed out."

        )

    except requests.exceptions.ConnectionError:

        return make_result(

            "UNKNOWN",

            "Unable to connect to VirusTotal."

        )

    except requests.exceptions.RequestException as e:

        print(
            "REQUEST ERROR:",
            e
        )

        return make_result(

            "UNKNOWN",

            "VirusTotal request failed."

        )

    except Exception as e:

        print(
            "VIRUSTOTAL ERROR:",
            e
        )

        return make_result(

            "UNKNOWN",

            "VirusTotal connection failed."

        )


# =========================================================
# URL ANALYZER API
# ML + VIRUSTOTAL + HEURISTIC
# =========================================================

@app.route(
    "/analyze-url",
    methods=["POST"]
)
def analyze_url():

    print("=" * 55)
    print("URL ANALYZER")
    print("=" * 55)

    try:

        data = request.get_json(
            silent=True
        ) or {}

        url = data.get(
            "url",
            ""
        )

        if not isinstance(
            url,
            str
        ):

            url = str(url)

        url = url.strip()

        print(
            "URL RECEIVED:",
            url
        )

        # =================================================
        # VALIDATE URL
        # =================================================

        normalized_url = normalize_url(
            url
        )

        if not normalized_url:

            return jsonify({

                "success": False,

                "error":
                    "Please enter a valid website URL."

            }), 400

        # =================================================
        # MACHINE LEARNING
        # =================================================

        print("=" * 55)
        print("MACHINE LEARNING ANALYSIS")
        print("=" * 55)

        ml_result = predict_url_with_ml(
            normalized_url
        )

        print(
            "ML STATUS:",
            ml_result["status"]
        )

        print(
            "ML SCAM PROBABILITY:",
            ml_result["scam_probability"],
            "%"
        )

        # =================================================
        # VIRUSTOTAL
        # =================================================

        print("=" * 55)
        print("VIRUSTOTAL ANALYSIS")
        print("=" * 55)

        vt_result = scan_url_with_virustotal(
            normalized_url
        )

        print(
            "VT STATUS:",
            vt_result.get(
                "status"
            )
        )

        # =================================================
        # LOCAL HEURISTIC
        # =================================================

        local_reasons = local_url_check(
            normalized_url
        )

        print(
            "LOCAL REASONS:",
            local_reasons
        )

        # =================================================
        # FINAL STATUS
        # =================================================

        vt_status = vt_result.get(
            "status",
            "UNKNOWN"
        )

        ml_status = ml_result.get(
            "status",
            "UNKNOWN"
        )

        final_status = "SAFE"

        final_message = (
            "No obvious security threat "
            "was detected."
        )

        # Highest priority:
        # VirusTotal malicious
        if vt_status == "RISK":

            final_status = "RISK"

            final_message = (
                "VirusTotal detected malicious "
                "activity. Avoid opening this URL."
            )

        # ML detects phishing
        elif ml_status == "RISK":

            final_status = "RISK"

            final_message = (
                "Machine Learning model detected "
                "this URL as potentially phishing "
                "or malicious."
            )

        # VirusTotal suspicious
        elif vt_status == "SUSPICIOUS":

            final_status = "SUSPICIOUS"

            final_message = (
                "VirusTotal detected suspicious "
                "activity. Verify the website "
                "before continuing."
            )

        # Local heuristic
        elif local_reasons:

            final_status = "SUSPICIOUS"

            final_message = (
                "Suspicious URL patterns were "
                "detected. Verify the website "
                "before continuing."
            )

        # VT unavailable but ML safe
        elif vt_status == "UNKNOWN":

            if ml_status == "SAFE":

                final_status = "SAFE"

                final_message = (
                    "Machine Learning model classified "
                    "this URL as potentially safe. "
                    "VirusTotal result is unavailable."
                )

            else:

                final_status = "UNKNOWN"

                final_message = (
                    "Unable to determine the URL "
                    "status completely."
                )

        # =================================================
        # FINAL RESPONSE
        # =================================================

        response_data = {

            "success": True,

            "url":
                normalized_url,

            # Final combined status
            "status":
                final_status,

            "message":
                final_message,

            # VirusTotal
            "malicious":
                int(
                    vt_result.get(
                        "malicious",
                        0
                    )
                    or 0
                ),

            "suspicious":
                int(
                    vt_result.get(
                        "suspicious",
                        0
                    )
                    or 0
                ),

            # Machine Learning
            "ml_status":
                ml_result.get(
                    "status",
                    "UNKNOWN"
                ),

            "ml_prediction":
                ml_result.get(
                    "prediction"
                ),

            "scam_probability":
                ml_result.get(
                    "scam_probability",
                    0
                ),

            # Local analysis
            "local_reasons":
                local_reasons,

            # Detailed result
            "result": {

                "url":
                    normalized_url,

                "status":
                    final_status,

                "message":
                    final_message,

                "malicious":
                    int(
                        vt_result.get(
                            "malicious",
                            0
                        )
                        or 0
                    ),

                "suspicious":
                    int(
                        vt_result.get(
                            "suspicious",
                            0
                        )
                        or 0
                    ),

                "ml_status":
                    ml_result.get(
                        "status",
                        "UNKNOWN"
                    ),

                "ml_prediction":
                    ml_result.get(
                        "prediction"
                    ),

                "scam_probability":
                    ml_result.get(
                        "scam_probability",
                        0
                    ),

                "local_reasons":
                    local_reasons

            }

        }

        # Add reasons
        if local_reasons:

            response_data["reasons"] = (
                local_reasons
            )

            response_data[
                "result"
            ]["reasons"] = (
                local_reasons
            )

        print("=" * 55)
        print("FINAL URL RESULT")
        print(
            "STATUS:",
            final_status
        )
        print(
            "ML:",
            ml_result
        )
        print(
            "VT:",
            vt_result
        )
        print("=" * 55)

        return jsonify(
            response_data
        )

    except Exception as e:

        print(
            "URL ANALYZER ERROR:",
            e
        )

        return jsonify({

            "success": False,

            "error":
                "URL analysis failed: "
                + str(e)

        }), 500


# =========================================================
# OPTIONAL URL FORM
# =========================================================

@app.route(
    "/analyze-url-form",
    methods=["POST"]
)
def analyze_url_form():

    url = request.form.get(
        "url",
        ""
    )

    normalized_url = normalize_url(
        url
    )

    if not normalized_url:

        return render_template(

            "qr_code.html",

            error=
                "Please enter a valid website URL."

        )

    # ML
    ml_result = predict_url_with_ml(
        normalized_url
    )

    # VirusTotal
    vt_result = scan_url_with_virustotal(
        normalized_url
    )

    reasons = local_url_check(
        normalized_url
    )

    # Final status
    if vt_result.get("status") == "RISK":

        final_status = "RISK"

    elif ml_result.get("status") == "RISK":

        final_status = "RISK"

    elif vt_result.get("status") == "SUSPICIOUS":

        final_status = "SUSPICIOUS"

    elif reasons:

        final_status = "SUSPICIOUS"

    else:

        final_status = "SAFE"

    return render_template(

        "qr_code.html",

        url=normalized_url,

        status=final_status,

        message=(
            vt_result.get(
                "message",
                ""
            )
        ),

        malicious=vt_result.get(
            "malicious",
            0
        ),

        suspicious=vt_result.get(
            "suspicious",
            0
        ),

        reasons=reasons,

        ml_status=ml_result.get(
            "status",
            "UNKNOWN"
        ),

        scam_probability=ml_result.get(
            "scam_probability",
            0
        )

    )


# =========================================================
# UPI ANALYSIS
# =========================================================

def analyze_upi(data):

    reasons = []

    score = 0

    parsed = urlparse(
        data
    )

    params = parse_qs(
        parsed.query
    )

    # =====================================================
    # UPI ID
    # =====================================================

    upi_id = unquote(

        params.get(
            "pa",
            [""]
        )[0]

    ).strip()

    # =====================================================
    # MERCHANT
    # =====================================================

    merchant = unquote(

        params.get(
            "pn",
            ["Unknown Merchant"]
        )[0]

    ).strip()

    # =====================================================
    # AMOUNT
    # =====================================================

    amount = params.get(
        "am",
        ["Not Specified"]
    )[0]

    if not amount or amount == "undefined":

        amount = "Not Specified"

    # =====================================================
    # CURRENCY
    # =====================================================

    currency = params.get(
        "cu",
        ["INR"]
    )[0]

    # =====================================================
    # UPI ID VALIDATION
    # =====================================================

    upi_pattern = (
        r"^[A-Za-z0-9._-]+"
        r"@[A-Za-z0-9._-]+$"
    )

    if not re.match(
        upi_pattern,
        upi_id
    ):

        score += 3

        reasons.append(
            "UPI ID format is unusual."
        )

    # =====================================================
    # AMOUNT VALIDATION
    # =====================================================

    if amount != "Not Specified":

        try:

            amount_value = float(
                amount
            )

            if amount_value <= 0:

                score += 2

                reasons.append(
                    "Payment amount is invalid."
                )

        except ValueError:

            score += 2

            reasons.append(
                "Payment amount format is invalid."
            )

    # =====================================================
    # SUSPICIOUS KEYWORDS
    # =====================================================

    suspicious_words = [

        "free",
        "gift",
        "winner",
        "prize",
        "claim",
        "reward",
        "bonus",
        "cashback",
        "lottery",
        "urgent",
        "verify",
        "verification",
        "refund",
        "lucky",
        "selected",
        "congratulations",
        "password",
        "otp",
        "login"

    ]

    combined_text = (

        merchant
        + " "
        + upi_id

    ).lower()

    for word in suspicious_words:

        if word in combined_text:

            score += 2

            reasons.append(

                "Suspicious keyword detected: "
                + word

            )

    # =====================================================
    # SUSPICIOUS MERCHANT
    # =====================================================

    suspicious_merchant_words = [

        "free",
        "gift",
        "winner",
        "prize",
        "reward",
        "lottery",
        "bonus",
        "cashback",
        "claim",
        "urgent"

    ]

    merchant_lower = (
        merchant.lower()
    )

    for word in suspicious_merchant_words:

        if word in merchant_lower:

            score += 3

            reasons.append(
                "Suspicious merchant name detected."
            )

            break

    # =====================================================
    # SUSPICIOUS PROVIDER
    # =====================================================

    suspicious_providers = [

        "fakebank",
        "unknownbank",
        "testbank",
        "fakeupi"

    ]

    provider = ""

    if "@" in upi_id:

        provider = (

            upi_id
            .split("@")[-1]
            .lower()

        )

    if provider in suspicious_providers:

        score += 4

        reasons.append(
            "Unknown or suspicious "
            "UPI provider detected."
        )

    # =====================================================
    # FINAL STATUS
    # =====================================================

    if score >= 8:

        status = "RISK"

        message = (

            "Multiple high-risk indicators "
            "were detected. Do not make the "
            "payment until the recipient is "
            "independently verified."

        )

    elif score >= 4:

        status = "SUSPICIOUS"

        message = (

            "Some suspicious indicators "
            "were detected. Verify the "
            "recipient before making the payment."

        )

    else:

        status = "SAFE"

        message = (

            "No obvious suspicious pattern "
            "was detected. Still verify the "
            "recipient before payment."

        )

    return {

        "merchant":
            merchant,

        "upi_id":
            upi_id,

        "amount":
            amount,

        "currency":
            currency,

        "status":
            status,

        "score":
            score,

        "message":
            message,

        "reasons":
            reasons,

        "payment_url":
            data

    }


# =========================================================
# QR IMAGE DECODER
# ONLY FOR UPI MODULE
# =========================================================

def decode_qr_image(file):

    image_bytes = file.read()

    if not image_bytes:

        return None

    image_array = np.frombuffer(

        image_bytes,

        np.uint8

    )

    image = cv2.imdecode(

        image_array,

        cv2.IMREAD_COLOR

    )

    if image is None:

        return None

    detector = cv2.QRCodeDetector()

    # First attempt
    data, points, _ = (
        detector.detectAndDecode(
            image
        )
    )

    if data:

        return data.strip()

    # Grayscale
    gray = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY

    )

    data, points, _ = (
        detector.detectAndDecode(
            gray
        )
    )

    if data:

        return data.strip()

    return None


# =========================================================
# UPI IMAGE SCAN
# =========================================================

@app.route(
    "/upi-scan",
    methods=["POST"]
)
def upi_scan():

    print("=" * 45)
    print("UPI QR IMAGE SCAN")
    print("=" * 45)

    if "upi_image" not in request.files:

        return render_template(

            "upi_payment.html",

            error=
                "UPI QR image was not uploaded."

        )

    file = request.files[
        "upi_image"
    ]

    if file.filename == "":

        return render_template(

            "upi_payment.html",

            error=
                "Please select a UPI QR image."

        )

    try:

        data = decode_qr_image(
            file
        )

        if not data:

            return render_template(

                "upi_payment.html",

                error=(

                    "UPI QR code was not detected. "
                    "Please upload a clear QR image."

                )

            )

        print(
            "UPI QR DATA:",
            data
        )

        if not data.lower().startswith(
            "upi://pay"
        ):

            return render_template(

                "upi_payment.html",

                error=(

                    "This is not a valid "
                    "UPI payment QR."

                )

            )

        result = analyze_upi(
            data
        )

        print(
            "UPI RESULT:",
            result
        )

        return render_template(

            "upi_payment.html",

            upi=result

        )

    except Exception as e:

        print(
            "UPI IMAGE ERROR:",
            e
        )

        return render_template(

            "upi_payment.html",

            error=str(e)

        )


# =========================================================
# UPI CAMERA SCAN
# =========================================================

@app.route(
    "/upi-camera-scan",
    methods=["POST"]
)
def upi_camera_scan():

    print("=" * 45)
    print("UPI CAMERA SCAN")
    print("=" * 45)

    try:

        body = request.get_json(
            silent=True
        ) or {}

        data = body.get(
            "data",
            ""
        )

        if not isinstance(
            data,
            str
        ):

            data = str(data)

        data = data.strip()

        print(
            "CAMERA UPI DATA:",
            data
        )

        if not data:

            return jsonify({

                "success": False,

                "error":
                    "UPI QR data was not detected."

            }), 400

        if not data.lower().startswith(
            "upi://pay"
        ):

            return jsonify({

                "success": False,

                "error": (

                    "This is not a valid "
                    "UPI payment QR."

                )

            }), 400

        result = analyze_upi(
            data
        )

        print(
            "UPI ID:",
            result["upi_id"]
        )

        print(
            "MERCHANT:",
            result["merchant"]
        )

        print(
            "AMOUNT:",
            result["amount"]
        )

        print(
            "SCORE:",
            result["score"]
        )

        print(
            "STATUS:",
            result["status"]
        )

        return jsonify({

            "success": True,

            "result": {

                "merchant":
                    result["merchant"],

                "upi_id":
                    result["upi_id"],

                "amount":
                    result["amount"],

                "currency":
                    result["currency"],

                "status":
                    result["status"],

                "score":
                    result["score"],

                "message":
                    result["message"],

                "reasons":
                    result["reasons"],

                "payment_url":
                    result["payment_url"]

            }

        })

    except Exception as e:

        print(
            "UPI CAMERA ERROR:",
            e
        )

        return jsonify({

            "success": False,

            "error":
                "Camera scan failed: "
                + str(e)

        }), 500


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )