import warnings
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from litscout.app import create_app
from litscout.config import Settings
from litscout.db import Store

warnings.filterwarnings("ignore", category=DeprecationWarning)
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def store() -> Store:
    return Store(":memory:")


@pytest.fixture
def settings() -> Settings:
    return Settings(sources_enabled="openalex,arxiv,crossref,demo", secret_key="test-secret", _env_file=None)


@pytest.fixture
def client(settings, store) -> TestClient:
    return TestClient(create_app(settings, store), base_url="http://testserver")


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")
