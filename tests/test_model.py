import joblib, pathlib, pandas as pd
from streamlit.testing.v1 import AppTest
from train import CAT, NUM

B = joblib.load("model/model.pkl")

def row(pkg):
    return pd.DataFrame([{"package": pkg, "package_type": "MOSEC", "price": 15000,
                          "duration_days": 30, "hour": 10, "dayofweek": 2, "month": 6}])

def test_proba_valid():
    p = B["pipeline"].predict_proba(row(B["packages"][0])[CAT + NUM])
    assert p.shape == (1, 2) and 0 <= p[0, 1] <= 1   # kontrak output

def test_unknown_package_tidak_crash():
    B["pipeline"].predict_proba(row("PAKET BARU")[CAT + NUM])  # kategori tak dikenal ditangani

def test_app_smoke():
    at = AppTest.from_file(str(pathlib.Path(__file__).parent.parent / "app.py")).run(timeout=30)
    assert not at.exception
    at.button[0].click().run(timeout=30)
    assert not at.exception and len(at.metric) == 1
