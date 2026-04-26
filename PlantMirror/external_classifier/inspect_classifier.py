import joblib
from pathlib import Path

pkl_path = "best_hai_classifier.pkl"
data = joblib.load(pkl_path)
print("Features in classifier:", len(data['features']))
print("First 10 features:", data['features'][:10])
print("Model type:", type(data['model']))
