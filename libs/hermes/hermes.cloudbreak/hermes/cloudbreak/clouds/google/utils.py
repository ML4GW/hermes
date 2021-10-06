from typing import Optional, Sequence, Union

from google.auth.transport.requests import Request as AuthRequest
from google.oauth2.service_account import Credentials as GoogleCredentials

Credentials = Union[str, GoogleCredentials, None]


def refresh(credentials: GoogleCredentials):
    credentials.refresh(AuthRequest())


def make_credentials(
    service_account_key_file: str,
    scopes: Optional[Sequence[str]] = None,
):
    """
    Cheap wrapper around service account creation
    class method to simplify a couple gotchas. Might
    either be overkill or may be better built as a
    class with more functionality, not sure yet.
    """
    scopes = scopes or ["https://www.googleapis.com/auth/cloud-platform"]
    credentials = GoogleCredentials.from_service_account_file(
        service_account_key_file,
        scopes=scopes,
    )
    refresh(credentials)
    return credentials
