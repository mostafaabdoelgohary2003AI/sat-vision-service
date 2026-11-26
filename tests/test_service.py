import io, json, os, sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_data_cache_loads():
    from src.data import load_arrays_cached, CLASSES
    X, y = load_arrays_cached()
    assert X.shape[1:] == (64, 64, 3)
    assert len(set(y.tolist())) == len(CLASSES)

@pytest.mark.skipif(not os.path.exists("results/metrics.json"), reason="metrics.json is produced by `python -m src.train` (hours on CPU); not run in CI")
def test_metrics_file_real():
    with open("results/metrics.json") as f:
        m = json.load(f)
    assert m["final_test_acc"] > 0.75
    assert m["train_images"] + m["test_images"] == 10800

@pytest.mark.skipif(not os.path.exists("results/model.pt"), reason="trained weights are not stored in git; run `python -m src.train` locally")
def test_api_predict(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DB", str(tmp_path / "t.db"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import numpy as np
    from PIL import Image
    from fastapi.testclient import TestClient
    from src import api
    from src.data import load_arrays_cached
    X, y = load_arrays_cached()
    img = Image.fromarray(X[0].astype("uint8"))
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    with TestClient(api.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/predict", files={"file": ("t.png", buf, "image/png")})
        assert r.status_code == 200
        preds = r.json()["predictions"]
        assert len(preds) == 3 and preds[0]["class"] in api.CLASSES
        assert len(client.get("/history").json()["predictions"]) >= 1
