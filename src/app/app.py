from flask import Flask, jsonify, request, send_from_directory
from sklearn.externals import joblib
app = Flask(__name__)

@app.route('/')
@app.route('/index.html')
def serve_static():
    return send_from_directory('/usr/src/app/public', 'index.html')
    
@app.route('/api/predictHousePrice', methods=["POST"])
def predict_house_price():
    j = request.get_json()
    lm = joblib.load('/usr/models/house-prices.pkl') 
    price = lm.predict([[j['squareFeet'],j['numBedrooms']]])[0][0]
    return jsonify({
        "ok": True,
        "price": price
    })