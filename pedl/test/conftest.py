import pytest
from unittest.mock import patch

@pytest.fixture
def mock_hydra_main():
    with patch('hydra.main', lambda x: x):
        yield
