import json

from google.auth import jwt

from src.settings import PUBSUB_KEY


def pubsub_credentials(publisher=False):
    """Create credentials from JSON file for Google Pub/Sub"""

    service_account_info = json.load(open(PUBSUB_KEY))
    credentials = jwt.Credentials.from_service_account_info(
        service_account_info,
        audience="https://pubsub.googleapis.com/google.pubsub.v1.Subscriber",
    )

    # The same for the publisher, except that the "audience" claim needs to be adjusted
    if publisher:
        publisher_audience = "https://pubsub.googleapis.com/google.pubsub.v1.Publisher"
        credentials = credentials.with_claims(audience=publisher_audience)

    return credentials
