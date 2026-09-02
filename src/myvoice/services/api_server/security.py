"""Auth, Host-header guard, and CORS for the local TTS API.

Three localhost-hardening controls (tech-spec Task 5, F9):

1. **Bearer auth** — when ``http_api_key`` is non-empty (the default, since a
   key is auto-generated on enable) every ``/v1/*`` request must present
   ``Authorization: Bearer <key>``; otherwise 401. Comparison is
   constant-time. An empty key is the explicit keyless opt-out.
2. **Host guard** — reject any request whose ``Host`` header host is not
   ``127.0.0.1``/``localhost`` with 400, to block DNS-rebinding/CSRF from a
   browser pointed at a name that resolves to the loopback.
3. **CORS** — restrictive default-deny (no wildcard credentialed origins).
   CORS alone does not stop a no-preflight POST from *reaching* the handler;
   the key + Host guard are the real protection.

The current key is read live from ``request.app.state.settings_provider()`` so
clearing/rotating the key in Settings takes effect without a server restart.
"""

from __future__ import annotations

import logging
import secrets

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _parse_host(host_header: str) -> str:
    """Extract the bare host from a Host header, handling IPv6 brackets.

    ``[::1]:7778`` -> ``::1``; ``127.0.0.1:7778`` -> ``127.0.0.1``;
    ``localhost`` -> ``localhost``. Returns lowercase; ``""`` if absent.
    """
    hp = (host_header or "").strip()
    if not hp:
        return ""
    if hp.startswith("["):
        # Bracketed IPv6 literal: take everything up to the closing bracket.
        end = hp.find("]")
        if end != -1:
            return hp[1:end].lower()
        return hp.lower()
    # IPv4/hostname: strip a trailing :port (a bare IPv6 without brackets is
    # not a valid Host header, so a lone ":" split is safe here).
    return (hp.rsplit(":", 1)[0] if ":" in hp else hp).lower()


def _current_api_key(request: Request) -> str:
    """Read the live API key from app state (empty string = keyless)."""
    provider = getattr(request.app.state, "settings_provider", None)
    if provider is None:
        return ""
    settings = provider()
    return getattr(settings, "http_api_key", "") or ""


async def verify_host(request: Request) -> None:
    """Reject requests whose Host header is not a loopback name (400)."""
    host_header = request.headers.get("host", "")
    host = _parse_host(host_header)
    if host not in _ALLOWED_HOSTS:
        logger.warning("Rejected request with non-loopback Host header: %r", host_header)
        raise HTTPException(status_code=400, detail="Invalid Host header")


async def verify_auth(request: Request) -> None:
    """Require a matching Bearer token unless the key is cleared (keyless)."""
    api_key = _current_api_key(request)
    if not api_key:
        # Keyless opt-out: the user explicitly cleared the key.
        return

    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=401,
            detail="Missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not secrets.compare_digest(token, api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def install_cors(app) -> None:
    """Attach a restrictive CORS policy (default-deny cross-origin)."""
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],          # deny all cross-origin by default
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def generate_api_key() -> str:
    """Generate a high-entropy URL-safe API key (tech-spec G12)."""
    return secrets.token_urlsafe(32)
