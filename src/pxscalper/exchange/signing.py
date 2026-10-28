"""PrimeXBT API v3 request signing.

PrimeXBT signs every authenticated call with **Ed25519**. The API key page gives
you a *key id* (``pxk_...``) and an Ed25519 *private key* in PKCS#8, base64.
The key id goes in a header; the private key signs the canonical string::

    METHOD
    /v3/trading/orders/market/place
    <query string exactly as sent, empty when none>
    1764580000123            <- epoch milliseconds
    <sha256(body_bytes) as lowercase hex>

The five lines are joined with ``\\n`` and no trailing newline. The signature is
base64 of the raw Ed25519 signature bytes.

Reference: https://api.primexbt.com/docs/trading.html#section/Authentication
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from dataclasses import dataclass, field

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import load_der_private_key

__all__ = ["ApiCredentials", "SignatureError", "canonical_string", "sign_request"]


class SignatureError(RuntimeError):
    """Raised when a credential cannot produce a valid signature."""


@dataclass
class ApiCredentials:
    """A PrimeXBT API key pair.

    ``key_id``      -- the public ``pxk_...`` identifier.
    ``secret_b64``  -- base64 PKCS#8 Ed25519 private key (shown once at creation).

    The secret may also be supplied raw (non-base64) or as PEM; we try the
    realistic encodings so a copy/paste mistake produces a helpful error instead
    of a confusing 401.
    """

    key_id: str
    secret_b64: str
    permission: str = "trade"  # "read" | "trade" (informational only)
    _key: Ed25519PrivateKey = field(init=False, repr=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not self.key_id or not self.key_id.startswith("pxk_"):
            raise SignatureError(
                f"PX_API_KEY looks wrong ({self.key_id!r}): PrimeXBT key ids start with 'pxk_'."
            )
        self._key = self._load(self.secret_b64)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _load(secret: str) -> Ed25519PrivateKey:
        secret = (secret or "").strip()
        if not secret:
            raise SignatureError(
                "PX_API_SECRET is empty. PrimeXBT shows the Ed25519 private key "
                "(PKCS#8, base64) exactly once when the key is created."
            )
        if "BEGIN" in secret:  # PEM
            from cryptography.hazmat.primitives.serialization import load_pem_private_key

            key = load_pem_private_key(secret.encode(), password=None)
        else:
            try:
                der = base64.b64decode(secret, validate=True)
            except Exception as exc:  # noqa: BLE001
                raise SignatureError(
                    "PX_API_SECRET is not valid base64. Note: a value starting with "
                    "'pxk_' is the *key id*, not the secret -- the secret is a long "
                    "base64 blob beginning with 'MC4CAQAwBQYDK2Vw...' (PKCS#8 Ed25519)."
                ) from exc
            try:
                key = load_der_private_key(der, password=None)
            except Exception as exc:  # noqa: BLE001
                raise SignatureError(
                    "PX_API_SECRET decoded from base64 but is not a PKCS#8 private "
                    "key. Recreate the API key on PrimeXBT and copy the secret as-is."
                ) from exc
        if not isinstance(key, Ed25519PrivateKey):
            raise SignatureError(
                f"PX_API_SECRET is a {type(key).__name__}, PrimeXBT requires Ed25519."
            )
        return key

    def sign(self, message: bytes) -> str:
        return base64.b64encode(self._key.sign(message)).decode()

    def __repr__(self) -> str:  # never leak the secret
        return f"ApiCredentials(key_id={self.key_id!r}, permission={self.permission!r})"


def canonical_string(
    method: str,
    path: str,
    query: str,
    timestamp_ms: int | str,
    body: bytes | None,
) -> str:
    """Build the five-line string PrimeXBT expects to be signed."""
    body_hash = hashlib.sha256(body or b"").hexdigest()
    return "\n".join([method.upper(), path, query or "", str(timestamp_ms), body_hash])


def sign_request(
    creds: ApiCredentials,
    method: str,
    path: str,
    query: str = "",
    body: bytes | None = None,
    timestamp_ms: int | None = None,
    recv_window_ms: int | None = None,
) -> dict[str, str]:
    """Return the three (or four) auth headers for one request.

    A fresh timestamp is used on every call: PrimeXBT accepts a signature for
    ``recvWindowMs`` (5000 ms) after its timestamp and at most 1 s before it, so
    a rejected request must be **re-signed**, never replayed verbatim.
    """
    ts = int(timestamp_ms if timestamp_ms is not None else time.time() * 1000)
    signed = canonical_string(method, path, query, ts, body)
    headers = {
        "X-API-KEY": creds.key_id,
        "X-API-TIMESTAMP": str(ts),
        "X-API-SIGNATURE": creds.sign(signed.encode()),
        "Accept": "application/json",
        "User-Agent": "pxscalper/0.1 (+https://api.primexbt.com/docs/trading.html)",
    }
    if body:
        headers["Content-Type"] = "application/json"
    if recv_window_ms:
        headers["X-API-RECV-WINDOW"] = str(int(recv_window_ms))
    return headers


def verify_hmac_webhook(secret: bytes, payload: bytes, signature: str) -> bool:
    """HMAC-SHA256 helper for out-of-band alerting webhooks (not used by PrimeXBT)."""
    digest = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, signature)
