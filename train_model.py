"""
Crop Recommendation System - Model Training
Maharashtra Agricultural ML Pipeline
"""
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 60)
print("  MAHARASHTRA CROP RECOMMENDATION - TRAINING")
print("=" * 60)

df = pd.read_csv('dataset/crop_recommendation_maharashtra_area_dataset.csv')
print(f"Dataset Shape: {df.shape}")
df.dropna(inplace=True)

district_encoder = LabelEncoder()
crop_encoder     = LabelEncoder()
df['district_encoded'] = district_encoder.fit_transform(df['district_area_maharashtra'])
df['crop_encoded']     = crop_encoder.fit_transform(df['crop_label'])

features = ['district_encoded','nitrogen','phosphorus','potassium','temperature','humidity','ph','rainfall']
X = df[features].values
y = df['crop_encoded'].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Random Forest':       RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    'Decision Tree':       DecisionTreeClassifier(max_depth=15, random_state=42),
    'SVM':                 SVC(probability=True, kernel='rbf', C=10, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=42),
}

results = {}
trained_models = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'Model':<25} {'Test Acc':>10} {'CV Mean':>10}")
print("-"*50)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cv     = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy', n_jobs=-1)
    results[name] = {
        'accuracy': round(float(acc)*100, 2),
        'cv_mean':  round(float(cv.mean())*100, 2),
        'cv_std':   round(float(cv.std())*100, 2),
        'classification_report': classification_report(y_test, y_pred,
                                   target_names=crop_encoder.classes_, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    trained_models[name] = model
    print(f"{name:<25} {acc*100:>9.2f}% {cv.mean()*100:>9.2f}%")

best_name  = max(results, key=lambda k: results[k]['accuracy'])
best_model = trained_models[best_name]
print(f"\nBest Model: {best_name} ({results[best_name]['accuracy']}%)")

feature_labels = ['District','Nitrogen','Phosphorus','Potassium','Temperature','Humidity','pH','Rainfall']
if hasattr(best_model,'feature_importances_'):
    imp = best_model.feature_importances_
else:
    rf_sur = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_sur.fit(X_train, y_train)
    imp = rf_sur.feature_importances_
feat_importance = {k: round(float(v),4) for k,v in zip(feature_labels,imp)}

crop_info = {
    'rice':        {'season':'Kharif','water':'High','fertilizer':'Urea + DAP','desc':'Staple cereal crop requiring flooded fields and high humidity.'},
    'cotton':      {'season':'Kharif','water':'Moderate','fertilizer':'NPK 20:20:0','desc':'Major cash crop of Vidarbha, needs deep black soil.'},
    'grapes':      {'season':'Rabi','water':'Moderate','fertilizer':'Nitrogen-rich','desc':'Premium horticultural crop grown extensively in Nashik.'},
    'banana':      {'season':'Perennial','water':'High','fertilizer':'Potassium-rich','desc':'Tropical fruit requiring warm temperatures and moisture.'},
    'mango':       {'season':'Summer','water':'Low','fertilizer':'NPK 10:26:26','desc':'King of fruits, grows well in Konkan and Marathwada.'},
    'orange':      {'season':'Rabi','water':'Moderate','fertilizer':'Boron + Zinc','desc':'Nagpur orange is world-famous, suited to Vidarbha region.'},
    'pomegranate': {'season':'Kharif','water':'Low','fertilizer':'Micronutrient mix','desc':'Drought-tolerant fruit, grown in Solapur and Nashik.'},
    'coconut':     {'season':'Perennial','water':'High','fertilizer':'NPK + FYM','desc':'Coastal crop thriving in humid Konkan conditions.'},
    'papaya':      {'season':'Kharif','water':'Moderate','fertilizer':'Nitrogen-rich','desc':'Fast-growing tropical fruit, sensitive to waterlogging.'},
    'watermelon':  {'season':'Summer','water':'Moderate','fertilizer':'Phosphorus-rich','desc':'Summer crop requiring sandy-loam soil and warm weather.'},
    'muskmelon':   {'season':'Summer','water':'Moderate','fertilizer':'NPK 19:19:19','desc':'Warm-season cucurbit grown in river-bed sandy soils.'},
    'maize':       {'season':'Kharif','water':'Moderate','fertilizer':'Urea + SSP','desc':'Versatile cereal for food, fodder, and industry.'},
    'chickpea':    {'season':'Rabi','water':'Low','fertilizer':'SSP + Rhizobium','desc':'Most important pulse crop, grown in Marathwada.'},
    'pigeonpeas':  {'season':'Kharif','water':'Low','fertilizer':'Phosphorus-rich','desc':'Tur dal, widely cultivated in rain-fed Maharashtra.'},
    'mungbean':    {'season':'Kharif','water':'Low','fertilizer':'Rhizobium inoculant','desc':'Short-duration pulse, excellent for soil health.'},
    'blackgram':   {'season':'Kharif','water':'Low','fertilizer':'SSP + Rhizobium','desc':'Urad dal, grown in Marathwada, tolerates moderate drought.'},
    'lentil':      {'season':'Rabi','water':'Low','fertilizer':'DAP + MOP','desc':'Cool-season pulse with high protein content.'},
    'kidneybeans': {'season':'Kharif','water':'Moderate','fertilizer':'NPK 60:80:40','desc':'Rajma, protein-rich, grown in Satara and Kolhapur.'},
    'mothbeans':   {'season':'Kharif','water':'Low','fertilizer':'Phosphorus','desc':'Drought-hardy legume for drier parts of Maharashtra.'},
    'coffee':      {'season':'Kharif','water':'High','fertilizer':'NPK + organic','desc':'Plantation crop requiring shade and well-drained slopes.'},
    'jute':        {'season':'Kharif','water':'High','fertilizer':'Nitrogen-rich','desc':'Natural fibre crop needing alluvial soil and high humidity.'},
    'apple':       {'season':'Rabi','water':'Moderate','fertilizer':'Calcium + Boron','desc':'Temperate fruit, grown at higher elevations in Maharashtra.'},
}

joblib.dump(best_model, 'crop_model.pkl')
joblib.dump({
    'district_encoder':    district_encoder,
    'crop_encoder':        crop_encoder,
    'scaler':              scaler,
    'feature_names':       feature_labels,
    'districts':           district_encoder.classes_.tolist(),
    'crops':               crop_encoder.classes_.tolist(),
    'model_results':       results,
    'best_model_name':     best_name,
    'feature_importance':  feat_importance,
    'crop_info':           crop_info,
    'dataset_stats':       df.describe().round(2).to_dict(),
    'crop_distribution':   df['crop_label'].value_counts().to_dict(),
    'district_distribution': df['district_area_maharashtra'].value_counts().to_dict(),
}, 'encoders.pkl')

print("Saved: crop_model.pkl & encoders.pkl")
