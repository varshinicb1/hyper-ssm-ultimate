"""
Shared test fixtures for the ICM test suite.
"""

import sys
import os
from typing import Generator

import pytest

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

# Ensure the project root is on sys.path so that `applications` and
# `hyper_ssm` can be imported regardless of where pytest was launched.
_project_root = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..")
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


@pytest.fixture(scope="session")
def hhm():
    """Shared HierarchicalHyperbolicMemory instance (session-scoped)."""
    from hyper_ssm.hierarchical_memory import HierarchicalHyperbolicMemory

    model = HierarchicalHyperbolicMemory(state_dim=32, num_scales=4)
    model.eval()
    return model


@pytest.fixture(scope="function")
def icm():
    """Fresh InfiniteContextMemory for each test."""
    from hyper_ssm.conversation_memory import InfiniteContextMemory

    torch.manual_seed(0)
    return InfiniteContextMemory(state_dim=32, num_scales=4)


@pytest.fixture(scope="function")
def server_client() -> Generator[TestClient, None, None]:
    """
    FastAPI TestClient against the ICM server.

    Patches LLM_BACKEND_AVAILABLE to False so the server uses the
    simulated chat fallback instead of loading a real model.
    """
    import applications.icm_server as srv

    # Force simulated fallback (no model loading needed)
    srv.LLM_BACKEND_AVAILABLE = False
    # Clear any residual state
    srv._session_meta.clear()
    srv._sim_counter = 0

    client = TestClient(srv.app)
    yield client

    srv._session_meta.clear()


@pytest.fixture(autouse=True)
def cleanup():
    """Reset shared mutable state after every test."""
    yield
    import applications.icm_server as srv

    srv._session_meta.clear()
    srv._sim_counter = 0


@pytest.fixture(scope="function")
def random_embedding() -> np.ndarray:
    """A random unit-normalised embedding vector (dim=32)."""
    np.random.seed(0)
    v = np.random.randn(32).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "unit: Fast unit tests that run in <1 s")
    config.addinivalue_line(
        "markers", "integration: Slower integration tests that exercise the server"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take several seconds to run"
    )
