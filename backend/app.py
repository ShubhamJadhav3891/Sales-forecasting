from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder='../frontend')

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    input_features = np.array([
        data['Store_ID'],
        data['Product_ID'],
        data['Promotion'],
        data['Holiday'],
        data['Day'],
        data['Month'],
        data['Year']
    ]).reshape(1, -1)

    prediction = model.predict(input_features)
    return jsonify({'predicted_sales': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)
