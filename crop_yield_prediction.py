import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load Dataset
data = pd.read_csv("crop_yield.csv")

# Encode categorical columns
le_crop = LabelEncoder()
data['Crop'] = le_crop.fit_transform(data['Crop'])

# Features and Target
X = data[['Rainfall', 'Temperature', 'Humidity',
          'Nitrogen', 'Phosphorus', 'Potassium', 'Crop']]

y = data['Yield']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Training
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Sample Prediction
sample = [[250, 28, 70, 90, 45, 50,
           le_crop.transform(['Rice'])[0]]]

prediction = model.predict(sample)

print("\nPredicted Crop Yield:", prediction[0], "quintals/hectare")