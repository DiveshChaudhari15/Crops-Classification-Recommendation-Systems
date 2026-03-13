# 🌾 CropAI Maharashtra — Crop Recommendation System

An end-to-end Machine Learning web application for crop recommendation across Maharashtra districts.

## Project Structure

```
crop-recommendation-system/
├── app.py                          # Flask web application
├── train_model.py                  # ML training pipeline
├── crop_model.pkl                  # Trained best model
├── encoders.pkl                    # Encoders, scaler, metadata
├── requirements.txt
├── CropRecommendation_Maharashtra_Complete.ipynb   # Full Jupyter Notebook
├── dataset/
│   └── crop_recommendation_maharashtra_area_dataset.csv
├── templates/
│   ├── base.html                   # Base layout
│   ├── index.html                  # Home page
│   ├── predict.html                # Prediction form
│   ├── result.html                 # Result page
│   ├── dashboard.html              # Analytics dashboard
│   ├── insights.html               # Dataset insights
│   └── about.html                  # Model performance + API docs
└── static/
    ├── css/
    └── js/
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python train_model.py

# 3. Run the Flask app
python app.py

# 4. Open browser
# http://localhost:5000
```

## Features

- **5 ML Models**: Random Forest, Decision Tree, SVM, Logistic Regression, Gradient Boosting
- **Hybrid Engine**: ML probabilities + Maharashtra agronomy rules
- **20 Districts**: All major Maharashtra districts supported
- **22 Crops**: Full crop coverage including cereals, pulses, fruits, cash crops
- **Top 5 Recommendations**: Ranked by confidence score
- **REST API**: `/api/predict` endpoint
- **Analytics Dashboard**: Chart.js visualizations
- **Dataset Insights**: Statistical summary, correlation heatmap, data table

## REST API

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "district": "Nashik",
    "nitrogen": 50,
    "phosphorus": 60,
    "potassium": 100,
    "temperature": 22,
    "humidity": 65,
    "ph": 6.2,
    "rainfall": 900
  }'
```

Response:
```json
{
  "recommended_crop": "grapes",
  "confidence": 18.4,
  "top5": [
    {"crop": "grapes", "confidence": 18.4},
    {"crop": "pomegranate", "confidence": 14.1},
    ...
  ]
}
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3, Flask |
| ML | Scikit-learn |
| Data | Pandas, NumPy |
| Persistence | Joblib |
| Frontend | Bootstrap 5, Chart.js |
| UI | Glassmorphism, CSS Gradients |

## Dataset

- **Source**: Maharashtra Crop Recommendation Dataset
- **Records**: 2,200
- **Districts**: 20
- **Crops**: 22
- **Features**: district, N, P, K, temperature, humidity, pH, rainfall
