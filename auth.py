"""Minimal staff authentication (shared PIN from the environment, constant-time comparison)."""
from __future__ import annotations

import hmac

import config

WEAK_PINS = {"", "change-me", "1234", "0000", "admin"}


def staff_login_enabled() -> bool:
    return bool(config.STAFF_PIN)


def pin_is_weak() -> bool:
    return config.STAFF_PIN.lower() in WEAK_PINS


def verify_staff(pin: str) -> bool:
    return staff_login_enabled() and hmac.compare_digest(pin.encode(), config.STAFF_PIN.encode())
