"""Latih model prediksi pembatalan (CANCEL) langganan Norton dari log transaksi cobrand."""
import sys, json, joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

CAT = ["package", "package_type"]
HORIZON = 35
NUM = ["price", "duration_days", "hour", "dayofweek", "month"]

def parse_dt(s):
    # format sumber: '20/7/2026 17:03:45.-1f+00' -> buang fraksi aneh, parse dayfirst
    s = s.astype(str).str.replace(r"\.-1f\+00$", "", regex=True).str.replace(r"\+00$", "", regex=True)
    return pd.to_datetime(s, dayfirst=True, errors="coerce")

def build_frame(xlsx):
    x = pd.ExcelFile(xlsx)
    c, pr, pk = (x.parse(n) for n in ["cobrandgen_transaction_log", "pricing", "package"])
    c = c[c.status_activated.isin(["ACTIVE", "CANCEL"])].copy()   # buang FAILED/PENDING (terlalu sedikit)
    # --- koreksi right-censoring ---
    # cancel rata-rata terjadi ~30 hari setelah mulai & data berakhir 7 Agu 2026, jadi transaksi baru
    # belum sempat cancel. Target = cancel dalam HORIZON hari; hanya pakai transaksi yang teramati penuh.
    start, canc = parse_dt(c.subscription_start_date), parse_dt(c.subscription_cancel_date)
    c["y"] = ((canc - start).dt.days <= HORIZON).astype(int)
    c = c[parse_dt(c.created_at) <= parse_dt(c.created_at).max() - pd.Timedelta(days=HORIZON)]
    pk = pk.assign(duration_days=pk.duration * pk.duration_type.map({"day": 1, "month": 30, "year": 365}).fillna(1))
    m = (c.merge(pr[["id", "package_id"]].rename(columns={"id": "pricing_id"}), on="pricing_id", how="left")
          .merge(pk[["id", "name", "duration_days"]].rename(columns={"id": "package_id", "name": "package"}), on="package_id", how="left"))
    t = parse_dt(m.created_at)
    # package_type: ambil kata kunci produk dari nama paket (360 / MOBILE SECURITY / dst.)
    m["package_type"] = m.package.str.extract(r"(360|MOBILE SECURITY|MOSEC|FAMILY|SECURITY)", expand=False).fillna("OTHER")
    m["hour"], m["dayofweek"], m["month"] = t.dt.hour, t.dt.dayofweek, t.dt.month
    # catatan: subscription_cancel_date/expired_at SENGAJA tidak dipakai (data leakage)
    return m[CAT + NUM + ["y"]], t

def make_pipeline():
    pre = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT)],
                            remainder="passthrough")
    clf = HistGradientBoostingClassifier(categorical_features=[0, 1], max_depth=4, learning_rate=0.05,
                                         max_iter=200, class_weight="balanced", random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])

if __name__ == "__main__":
    xlsx = sys.argv[1] if len(sys.argv) > 1 else "data/Data_Transaksi_Norton.xlsx"
    df, t = build_frame(xlsx)
    X, y = df[CAT + NUM], df.y
    # split berbasis waktu (bukan acak) supaya mirip kondisi produksi
    cut = t.loc[df.index].quantile(0.8)
    tr, te = t.loc[df.index] <= cut, t.loc[df.index] > cut
    pipe = make_pipeline().fit(X[tr], y[tr])
    p = pipe.predict_proba(X[te])[:, 1]
    base = y[te].mean()
    print(f"baseline cancel rate={base:.3f}  AUC={roc_auc_score(y[te], p):.3f}  F1={f1_score(y[te], p>0.5):.3f}")
    print(classification_report(y[te], p > 0.5, digits=3))
    pipe.fit(X, y)  # fit ulang di semua data untuk model produksi
    joblib.dump({"pipeline": pipe, "packages": sorted(df.package.dropna().unique().tolist()),
                 "types": sorted(df.package_type.unique().tolist()),
                 "durations": df.groupby("package").duration_days.first().to_dict(),
                 "prices": df.groupby("package").price.median().to_dict()}, "model/model.pkl")
    json.dump({"auc": float(roc_auc_score(y[te], p)), "cancel_rate": float(y.mean()), "n": int(len(df))}, open("model/metrics.json", "w"))
