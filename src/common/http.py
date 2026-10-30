"""HTTP GET with timeout and retries, shared by the API clients."""
from __future__ import annotations

import logging
import time

import requests

log = logging.getLogger(__name__)

HEADERS = {"Accept": "application/json, application/prs.coverage+json"}


def get_json(url: str, params: dict | None = None, retries: int = 4, timeout: int = 60):
    """Return the parsed JSON, or None on 404 (COSMOS uses 404 for "no data").

    Retries with exponential backoff (1, 2, 4 s) because the APIs sometimes
    time out; after the last try the error is raised so the job fails loudly
    instead of writing incomplete data.
    """
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == retries - 1:
                raise RuntimeError(f"GET {url} failed after {retries} tries: {e}") from e
            log.warning("GET %s failed (%s), retrying", url, e)
            time.sleep(2**attempt)
