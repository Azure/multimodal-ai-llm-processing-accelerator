import logging
import os

import requests

logger = logging.getLogger(__name__)


def check_if_env_var_is_set(env_var_name: str) -> bool:
    """
    Check that an environment variable has a value and that it has been
    correctly set (and isn't the default instructional value to copy the
    value from the AZD environment outputs.)
    """
    return os.getenv(env_var_name) and not os.getenv(env_var_name).startswith("Copy ")


def check_if_azurite_storage_emulator_is_running() -> bool:
    """
    Check if the Azurite storage emulator is running. This should fail
    if the HTTP request cannot create a connection, while any other
    response or exceptions are treated as though the emulator is running.
    """
    # Attempt request to Azurite storage emulator
    try:
        requests.get("http://127.0.0.1:10000/")
    except requests.exceptions.ConnectionError as _e:
        return False
    return True
