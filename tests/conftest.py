import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.update({
    "ENV": "test", "DATABASE_URL": f"sqlite:///{_tmp}/test.db", "DATA_DIR": f"{_tmp}/data",
    "AUTO_CREATE_TABLES": "true", "API_KEY": "", "RATE_LIMIT_PER_MINUTE": "0",
})

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services import processing  # noqa: E402

PDF = b"%PDF-1.4\n% fake pdf for tests\n"


@pytest.fixture()
def client(monkeypatch):
    submitted = []
    monkeypatch.setattr(processing, "submit", submitted.append)
    with TestClient(app) as c:
        c.submitted = submitted
        yield c
