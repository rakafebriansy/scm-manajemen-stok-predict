from flask import Flask, request, jsonify
import os
from predict import predict_next_days

app = Flask(__name__)

def get_available_products():
    return sorted(set(f.split('_model.h5')[0] for f in os.listdir('saved_models') if f.endswith('_model.h5')))

@app.route('/predict', methods=['GET'])
def predict():
    produk = request.args.get('produk', '').lower().strip()
    days = int(request.args.get('days', 7))

    available_products = get_available_products()
    if produk not in available_products:
        return jsonify({'error': f"Produk '{produk}' tidak ditemukan. Pilihan tersedia: {available_products}"}), 404

    try:
        result = predict_next_days(produk, days)
        return jsonify({
            'produk': produk,
            'prediksi': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return jsonify({'message': 'API Prediksi Penjualan Produk Siap Pakai!', 'produk_tersedia': get_available_products()})

if __name__ == '__main__':
    app.run(debug=True)
