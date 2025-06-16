import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

def predict_next_days(produk, days=7):
    model_path = f"saved_models/{produk}_model.h5"
    meta_path = f"saved_models/{produk}_meta.pkl"

    model = load_model(model_path)
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    scaler = meta['scaler']
    time_step = meta['time_step']
    smoothed = meta['smoothed']
    data_asli = meta['data_asli']

    # Preprocess
    scaled_data = scaler.transform(smoothed.to_frame())
    last_window = scaled_data[-time_step:].reshape(1, time_step, 1)

    # Predict next N days
    predictions = []
    input_seq = last_window.copy()

    for _ in range(days):
        pred = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(pred)

        # Optional: Smoothing prediction
        smoothed_pred = np.mean(predictions[-3:]) if len(predictions) >= 3 else pred
        input_seq = np.concatenate((input_seq[:, 1:, :], [[[smoothed_pred]]]), axis=1)

    # Inverse transform
    future_pred = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # Generate dates
    last_date = smoothed.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    result = {date: float(pred) for date, pred in zip(future_dates.strftime('%Y-%m-%d'), np.round(future_pred, 2))}

    return result
