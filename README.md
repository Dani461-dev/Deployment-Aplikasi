# Norton Churn Predictor
Prediksi peluang langganan Norton (cobrand Telkomsel) dibatalkan ≤ 35 hari.
- Latih ulang: `python train.py data/Data_Transaksi_Norton.xlsx` (data mentah TIDAK di-commit: berisi customer_id/IP)
- Jalankan: `streamlit run app.py` · Tes: `pytest -q` · Docker: `docker build -t norton-churn .`
- Deploy: share.streamlit.io → repo ini → `app.py`
