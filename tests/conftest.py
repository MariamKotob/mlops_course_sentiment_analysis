import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlops_practitioner_course.api import app
from mlops_practitioner_course.config import Settings


class MockModelServiceToAsync:
    async def predict(self, texts):
        return [0.9] * len(texts)
    async def get_threshold(self):
        return 0.5

class MockModelService:
    def __init__(self):
        self.to_async = MockModelServiceToAsync()

@pytest.fixture
def sample_reviews():
    return ["المنتج رائع جدا", "سيء ولا أنصح به", "عادي"]

@pytest.fixture
def client(monkeypatch):
    from mlops_practitioner_course.api import get_model_service
    # Override the get_model_service dependency
    app.dependency_overrides[get_model_service] = lambda: MockModelService()
    
    # Default Settings, not config.yaml, so tests don't depend on the local config.
    monkeypatch.setattr(app.state, "settings", Settings(), raising=False)
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up overrides
    app.dependency_overrides.clear()
