from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _mock_rag_chain():
    """Тесты не поднимают реальные Qdrant и LLM: lifespan получает мок-цепочку."""
    with patch("app.main.build_rag_chain") as build:
        build.return_value = (MagicMock(), MagicMock())
        yield build
