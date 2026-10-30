"""Load the pipeline configuration.

Two sources, with no key in both:
- config/settings.yaml: the pipeline definition (station, API URLs, feature version), versioned with the code
- .env / environment: deployment values (STORAGE_URI), so Docker and CI can set them

Everything a pipeline needs to know comes from here, so nothing is hard-coded in the code.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

DEFAULT_SETTINGS = Path("config/settings.yaml")


class Station(BaseModel):
    id: str
    name: str
    latitude: float
    longitude: float


class Cosmos(BaseModel):
    base_url: str
    collection: str
    parameters: list[str]
    backfill_start: date
    refetch_days: int


class OpenMeteo(BaseModel):
    previous_runs_url: str
    archive_start: date


class Features(BaseModel):
    version: str
    issue_gap: int


class Config(BaseSettings):
    # From the environment (required, there is no default location).
    storage_uri: str = Field(min_length=1)
    # From settings.yaml.
    station: Station
    cosmos: Cosmos
    openmeteo: OpenMeteo
    features: Features
    ranges: dict[str, tuple[float, float]]


def load_config(path: Path | str | None = None) -> Config:
    # .env fills os.environ (existing variables win), which also makes credentials
    # such as GOOGLE_APPLICATION_CREDENTIALS visible to the libraries that read them.
    load_dotenv(find_dotenv(usecwd=True))
    with open(path or DEFAULT_SETTINGS) as f:
        return Config(**yaml.safe_load(f))
