"""
Crop Recommendation System - Flask Backend
Maharashtra Agricultural AI System
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import os

app = Flask(__name__)

# ── Load model artifacts ──────────────────────────────────
model   = joblib.load('crop_model.pkl')
enc     = joblib.load('encoders.pkl')

district_encoder  = enc['district_encoder']
crop_encoder      = enc['crop_encoder']
scaler            = enc['scaler']
districts         = enc['districts']
crops             = enc['crops']
model_results     = enc['model_results']
best_model_name   = enc['best_model_name']
feat_importance   = enc['feature_importance']
crop_info         = enc['crop_info']
dataset_stats     = enc['dataset_stats']
crop_dist         = enc['crop_distribution']
district_dist     = enc['district_distribution']

# ── District-crop affinity (real Maharashtra agricultural knowledge) ─
DISTRICT_AFFINITY = {
    'Nashik':      ['grapes','onion','tomato','wheat','maize','pomegranate'],
    'Pune':        ['sugarcane','grapes','wheat','maize','rice','pomegranate'],
    'Nagpur':      ['orange','cotton','rice','soybean','wheat'],
    'Kolhapur':    ['sugarcane','rice','banana','coconut','blackgram'],
    'Sangli':      ['sugarcane','grapes','turmeric','banana','cotton'],
    'Satara':      ['sugarcane','rice','strawberry','maize','soybean'],
    'Ahmednagar':  ['sugarcane','grapes','maize','wheat','onion','pomegranate'],
    'Solapur':     ['pomegranate','cotton','soybean','wheat','onion'],
    'Aurangabad':  ['cotton','soybean','maize','orange','pomegranate'],
    'Latur':       ['soybean','cotton','tur','wheat','maize','grapes'],
    'Osmanabad':   ['soybean','cotton','tur','wheat','banana'],
    'Nanded':      ['soybean','cotton','sugarcane','maize','pigeonpeas'],
    'Amravati':    ['cotton','soybean','orange','wheat','rice'],
    'Akola':       ['cotton','soybean','wheat','chickpea','sorghum'],
    'Wardha':      ['cotton','soybean','wheat','orange','rice'],
    'Beed':        ['cotton','soybean','sugarcane','banana','pomegranate'],
    'Jalgaon':     ['banana','cotton','wheat','maize','chickpea'],
    'Dhule':       ['banana','wheat','maize','cotton','soybean'],
    'Bhandara':    ['rice','wheat','soybean','cotton','maize'],
    'Parbhani':    ['cotton','soybean','pigeonpeas','mungbean','wheat'],
}

def get_top5_crops(district, nitrogen, phosphorus, potassium,
                   temperature, humidity, ph, rainfall):
    """Hybrid recommendation: ML probabilities + agronomy rules."""
    # ML prediction
    d_enc  = district_encoder.transform([district])[0]
    feats  = np.array([[d_enc, nitrogen, phosphorus, potassium,
                         temperature, humidity, ph, rainfall]])
    scaled = scaler.transform(feats)
    proba  = model.predict_proba(scaled)[0]

    # Boost based on agronomy rules
    boosted = proba.copy()
    affinity = DISTRICT_AFFINITY.get(district, [])
    for c in affinity:
        if c in crop_encoder.classes_:
            idx = list(crop_encoder.classes_).index(c)
            boosted[idx] *= 2.5

    # Agronomy feature boosts
    for i, crop in enumerate(crop_encoder.classes_):
        info = crop_info.get(crop, {})
        water = info.get('water', 'Moderate')
        if water == 'High'     and rainfall > 800: boosted[i] *= 1.4
        if water == 'Low'      and rainfall < 500: boosted[i] *= 1.4
        if water == 'Moderate' and 500 <= rainfall <= 800: boosted[i] *= 1.2
        if crop in ['rice','jute','coconut'] and humidity > 70: boosted[i] *= 1.3
        if crop in ['cotton','mothbeans','pomegranate'] and temperature > 28: boosted[i] *= 1.2
        if crop in ['apple','lentil','chickpea'] and temperature < 25: boosted[i] *= 1.2
        if crop in ['chickpea','mungbean','blackgram','lentil'] and nitrogen < 50: boosted[i] *= 1.2
        if crop in ['grapes','banana','papaya'] and potassium > 100: boosted[i] *= 1.2

    # Normalise & rank
    total  = boosted.sum()
    norm   = boosted / total if total > 0 else boosted
    top5_idx = np.argsort(norm)[::-1][:5]
    return [(crop_encoder.classes_[i], round(float(norm[i])*100, 1)) for i in top5_idx]

# ═════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html', districts=districts)

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html', districts=districts)

    data = request.form if request.form else request.json
    try:
        district    = data.get('district','Nashik')
        nitrogen    = float(data.get('nitrogen', 50))
        phosphorus  = float(data.get('phosphorus', 60))
        potassium   = float(data.get('potassium', 40))
        temperature = float(data.get('temperature', 25))
        humidity    = float(data.get('humidity', 70))
        ph          = float(data.get('ph', 6.5))
        rainfall    = float(data.get('rainfall', 800))

        top5 = get_top5_crops(district, nitrogen, phosphorus, potassium,
                               temperature, humidity, ph, rainfall)

        best_crop, confidence = top5[0]
        info = crop_info.get(best_crop, {
            'season':'Kharif','water':'Moderate',
            'fertilizer':'NPK balanced','desc':'Suitable for your conditions.'
        })

        if request.is_json:
            return jsonify({'recommended_crop': best_crop, 'confidence': confidence,
                            'top5': [{'crop':c,'confidence':conf} for c,conf in top5]})

        return render_template('result.html',
            crop=best_crop, confidence=confidence, top5=top5,
            info=info, district=district,
            nitrogen=nitrogen, phosphorus=phosphorus, potassium=potassium,
            temperature=temperature, humidity=humidity, ph=ph, rainfall=rainfall)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html',
        crop_dist=json.dumps(crop_dist),
        district_dist=json.dumps(district_dist),
        feat_importance=json.dumps(feat_importance))

@app.route('/insights')
def insights():
    df = pd.read_csv('dataset/crop_recommendation_maharashtra_area_dataset.csv')
    table_html = df.head(50).to_html(classes='table table-sm table-striped', index=False, border=0)
    stats_html = df.describe().round(2).to_html(classes='table table-sm table-bordered', border=0)
    corr = df.drop(columns=['district_area_maharashtra','crop_label']).corr().round(3)
    corr_json = corr.to_dict()
    corr_cols  = corr.columns.tolist()
    corr_data  = corr.values.tolist()
    return render_template('insights.html',
        table_html=table_html, stats_html=stats_html,
        corr_json=json.dumps(corr_data), corr_cols=json.dumps(corr_cols),
        total_rows=len(df), total_features=len(df.columns)-1)

@app.route('/about')
def about():
    return render_template('about.html', model_results=model_results,
        best_model_name=best_model_name, feat_importance=feat_importance)

# ── JSON API ─────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    try:
        top5 = get_top5_crops(
            data['district'], data['nitrogen'], data['phosphorus'],
            data['potassium'], data['temperature'], data['humidity'],
            data['ph'], data['rainfall']
        )
        return jsonify({'recommended_crop': top5[0][0], 'confidence': top5[0][1],
                        'top5': [{'crop':c,'confidence':conf} for c,conf in top5]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
