import os
import pytest


@pytest.fixture(autouse=True)
def hermetic_docker_env(monkeypatch):
  """Prevents unit tests from interacting with host Docker daemon unless explicitly testing live Docker."""
  if not os.environ.get("CHATTY_TEST_DOCKER_LIVE"):
    monkeypatch.setattr("chatty.backends.is_docker_available", lambda docker_bin="docker", docker_host=None: False)
