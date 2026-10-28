"""PrimeXBT request signing must be byte-exact or every call fails with 401."""

from __future__ import annotations

import base64
import hashlib

import pytest

from pxscalper.exchange.signing import (
    ApiCredentials,
    SignatureError,
    canonical_string,
    sign_request,
)


def _make_creds() -> ApiCredentials:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption

    key = Ed25519PrivateKey.generate()
    der = key.private_bytes(Encoding.DER, PrivateFormat.PKCS8, NoEncryption())
    return ApiCredentials("pxk_testkey123", base64.b64encode(der).decode())


def test_canonical_string_is_five_lines():
    body = b'{"symbol":"EUR/USD"}'
    signed = canonical_string("POST", "/v3/trading/orders/market/place", "", 1764580000123, body)
    lines = signed.split("\n")
    assert lines[0] == "POST"
    assert lines[1] == "/v3/trading/orders/market/place"
    assert lines[2] == ""
    assert lines[3] == "1764580000123"
    assert lines[4] == hashlib.sha256(body).hexdigest()
    assert not signed.endswith("\n")


def test_empty_body_hashes_empty_string():
    signed = canonical_string("GET", "/v3/trading/metrics", "", 1, None)
    assert signed.split("\n")[4] == hashlib.sha256(b"").hexdigest()


def test_sign_request_headers_and_determinism():
    creds = _make_creds()
    body = b'{"symbol":"EUR/USD","side":"BUY","qty":0.1}'
    h1 = sign_request(creds, "POST", "/v3/trading/markets", "", body, timestamp_ms=1764580000123)
    h2 = sign_request(creds, "POST", "/v3/trading/markets", "", body, timestamp_ms=1764580000123)
    assert h1["X-API-KEY"] == "pxk_testkey123"
    assert h1["X-API-TIMESTAMP"] == "1764580000123"
    assert h1["X-API-SIGNATURE"] == h2["X-API-SIGNATURE"]
    assert h1["Content-Type"] == "application/json"
    # base64 of a 64-byte Ed25519 signature
    assert len(base64.b64decode(h1["X-API-SIGNATURE"])) == 64


def test_different_timestamp_changes_signature():
    creds = _make_creds()
    h1 = sign_request(creds, "GET", "/v3/trading/metrics", "", None, timestamp_ms=1)
    h2 = sign_request(creds, "GET", "/v3/trading/metrics", "", None, timestamp_ms=2)
    assert h1["X-API-SIGNATURE"] != h2["X-API-SIGNATURE"]


def test_key_id_used_as_secret_gives_actionable_error():
    with pytest.raises(SignatureError) as exc:
        ApiCredentials("pxk_examplekeynotsecret", "pxk_examplekeynotsecret")
    message = str(exc.value)
    assert "key id" in message.lower() or "not the secret" in message.lower()


def test_bad_prefix_rejected():
    with pytest.raises(SignatureError):
        ApiCredentials("not-a-key", "MC4CAQAw")


def test_secret_never_in_repr():
    creds = _make_creds()
    assert "MC4" not in repr(creds)
    assert "pxk_testkey123" in repr(creds)
