"""SecretsManager — OS-level secrets via Python keyring.

All secret storage is delegated to ``keyring`` (OS Keychain on Mac,
``libsecret`` on WSL2 Dell). No plaintext secrets are written to disk.
"""
from __future__ import annotations

import logging

import keyring
import keyring.errors

from rbi_core.exceptions import SecretsNotFoundError

logger = logging.getLogger(__name__)


class SecretsManager:
    """Static helpers for OS-keyring-backed secret management."""

    @staticmethod
    def get_secret(service: str, key: str) -> str:
        """Retrieve a secret from the OS keyring.

        Args:
            service: Keyring service namespace (e.g. ``"rbi_swarm"``).
            key:     Secret identifier (e.g. ``"exchange_api_key"``).

        Returns:
            The stored secret value as a plain string.

        Raises:
            SecretsNotFoundError: If no secret is stored for the given
                service/key combination.
        """
        value = keyring.get_password(service, key)
        if value is None:
            raise SecretsNotFoundError(
                f"Secret not found in OS keyring: service={service!r}, key={key!r}"
            )
        return value

    @staticmethod
    def set_secret(service: str, key: str, value: str) -> None:
        """Store a secret in the OS keyring.

        The secret *value* is NEVER written to any log.

        Args:
            service: Keyring service namespace.
            key:     Secret identifier.
            value:   Secret value to store.
        """
        keyring.set_password(service, key, value)
        logger.info(
            "secret_stored",
            extra={
                "event": "secret_stored",
                "service": service,
                "key": key,
            },
        )

    @staticmethod
    def delete_secret(service: str, key: str) -> None:
        """Delete a secret from the OS keyring.

        Does not raise if the secret does not exist.

        Args:
            service: Keyring service namespace.
            key:     Secret identifier.
        """
        try:
            keyring.delete_password(service, key)
        except keyring.errors.PasswordDeleteError:
            logger.debug(
                "secret_delete_noop",
                extra={
                    "event": "secret_delete_noop",
                    "service": service,
                    "key": key,
                    "context": "Secret not found; delete skipped",
                },
            )
