import json, joblib, pandas as pd, streamlit as st
from train import CAT, NUM

st.set_page_config(page_title="Norton Churn Predictor", page_icon="🛡️")

@st.cache_resource  # model dimuat sekali, bukan tiap interaksi
def load():
    return joblib.load("model/model.pkl"), json.load(open("model/metrics.json"))

bundle, metrics = load()
pipe = bundle["pipeline"]

def predict_df(df: pd.DataFrame) -> pd.Series:
    return pd.Series(pipe.predict_proba(df[CAT + NUM])[:, 1], index=df.index, name="p_cancel_35d")

st.title("🛡️ Prediksi Pembatalan Langganan Norton")
st.caption(f"Peluang pelanggan cancel ≤ 35 hari · AUC uji (split waktu) = {metrics['auc']:.2f} · n = {metrics['n']:,}")

tab1, tab2 = st.tabs(["Satu transaksi", "Batch CSV"])

with tab1:
    pkg = st.selectbox("Paket", bundle["packages"])
    price = st.number_input("Harga (Rp)", min_value=0, value=int(bundle["prices"].get(pkg, 15000)), step=1000)
    c1, c2, c3 = st.columns(3)
    hour = c1.slider("Jam beli", 0, 23, 12)
    dow = c2.selectbox("Hari", range(7), format_func=lambda i: ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"][i])
    month = c3.selectbox("Bulan", range(1, 13), index=5)
    if st.button("Prediksi", type="primary"):
        ptype = next((t for t in ["360", "MOBILE SECURITY", "MOSEC", "FAMILY", "SECURITY"] if t in pkg), "OTHER")
        row = pd.DataFrame([{"package": pkg, "package_type": ptype, "price": price,
                             "duration_days": bundle["durations"].get(pkg, 30),
                             "hour": hour, "dayofweek": dow, "month": month}])
        p = float(predict_df(row).iloc[0])
        st.metric("Peluang cancel ≤ 35 hari", f"{p:.1%}")
        (st.error if p >= 0.5 else st.success)("Risiko tinggi" if p >= 0.5 else "Risiko rendah")

with tab2:
    st.write("Kolom wajib: `" + "`, `".join(CAT + NUM) + "`")
    f = st.file_uploader("Unggah CSV", type="csv")
    if f:
        d = pd.read_csv(f)
        miss = set(CAT + NUM) - set(d.columns)
        if miss:
            st.error(f"Kolom kurang: {sorted(miss)}")
        else:
            out = d.assign(p_cancel_35d=predict_df(d))
            st.dataframe(out)
            st.download_button("Unduh hasil", out.to_csv(index=False), "hasil.csv")
