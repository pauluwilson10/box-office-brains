import pandas as pd
import numpy as json
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.cluster import KMeans

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv('tmdb_5000_movies.csv')

# Drop rows with missing budget or revenue
df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
print(f"Working with {len(df)} valid movies after filtering")

# Feature engineering
print("Engineering features...")
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_month'] = df['release_date'].dt.month.fillna(0).astype(int)
df['release_year'] = df['release_date'].dt.year.fillna(0).astype(int)
df['num_genres'] = df['genres'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)

# Extract production companies count instead of cast
df['production_companies_count'] = df['production_companies'].apply(
    lambda x: len(eval(x)) if pd.notnull(x) else 0
)

# Extract keyword count as a feature
df['keywords_count'] = df['keywords'].apply(
    lambda x: len(eval(x)) if pd.notnull(x) else 0
)

# Calculate popularity to vote ratio
df['popularity_vote_ratio'] = df['popularity'] / df['vote_count'].clip(lower=1)

# Select features
features = df[[
    'budget', 
    'runtime', 
    'release_month', 
    'release_year',
    'num_genres',
    'production_companies_count',
    'keywords_count',
    'popularity',
    'vote_average',
    'vote_count'
]].fillna(0)

target = df['revenue']

# Scale features
print("Scaling features and training models...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Model R² score - Training: {train_score:.4f}, Test: {test_score:.4f}")

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Save models
print("Saving models...")
joblib.dump(model, 'models/revenue_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(kmeans, 'models/clustering_model.pkl')

# Save feature list for reference by the app
feature_list = list(features.columns)
joblib.dump(feature_list, 'models/feature_list.pkl')

print("✅ Models trained and saved in /models/")
