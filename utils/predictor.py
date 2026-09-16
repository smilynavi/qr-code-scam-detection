from urllib.parse import urlparse


def check_url(url):
    """
    Temporary rule-based URL checker.
    This will later be replaced by the Machine Learning model.
    """

    if not url:
        return "UNKNOWN"

    url = url.strip().lower()

    parsed = urlparse(url)

    suspicious_words = [
        "login",
        "verify",
        "verification",
        "account",
        "password",
        "free-gift",
        "claim-prize",
        "urgent",
        "secure-bank"
    ]

    suspicious_extensions = [
        ".xyz",
        ".tk",
        ".top",
        ".click"
    ]

    # Check for suspicious words
    for word in suspicious_words:
        if word in url:
            return "SUSPICIOUS"

    # Check for suspicious domain extensions
    for extension in suspicious_extensions:
        if extension in parsed.netloc:
            return "SUSPICIOUS"

    # HTTP is less secure than HTTPS
    if parsed.scheme == "http":
        return "SUSPICIOUS"

    # HTTPS URLs without the above indicators
    if parsed.scheme == "https":
        return "SAFE"

    return "SUSPICIOUS"