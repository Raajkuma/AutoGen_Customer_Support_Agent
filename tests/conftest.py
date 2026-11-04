import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Every test gets its own database, ticket log, traces and index directory."""
    for name, rel in {"DB_FILE": "shop.db", "LOG_FILE": "tickets.csv", "TRACE_DIR": "traces",
                      "INDEX_DIR": "index", "APP_LOG": "app.log"}.items():
        monkeypatch.setattr(config, name, tmp_path / rel)
    monkeypatch.setattr(config, "EMBED_ENABLED", False)
    import store, tickets
    store.init_db()
    tickets.init()
