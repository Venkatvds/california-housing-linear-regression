import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature names from California Housing dataset
feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", 
                 "Population", "AveOccup", "Latitude", "Longitude"]

# Display model coefficients
print("=" * 60)
print("LINEAR REGRESSION MODEL COEFFICIENTS")
print("=" * 60)
print(f"\nIntercept (β₀): {model.intercept_:.4f}")
print("\nFeature Coefficients:")
print("-" * 40)

for name, coef in zip(feature_names, model.coef_):
    print(f"  {name:15s}: {coef:+.4f}")

# Show which features have the most impact
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (by absolute coefficient value)")
print("=" * 60)
coef_importance = sorted(zip(feature_names, model.coef_), 
                         key=lambda x: abs(x[1]), reverse=True)
for i, (name, coef) in enumerate(coef_importance, 1):
    print(f"  {i}. {name:15s}: {abs(coef):.4f}")

# Demonstrate a prediction
print("\n" + "=" * 60)
print("DEMONSTRATION: Making a Prediction")
print("=" * 60)

# Sample input (typical California neighborhood)
sample_input = pd.DataFrame([{
    "MedInc": 3.5,      # $35,000 median income
    "HouseAge": 28,     # 28 years old
    "AveRooms": 5.0,    # 5 rooms per household
    "AveBedrms": 1.0,   # 1 bedroom
    "Population": 1400, # Block population
    "AveOccup": 3.0,   # 3 people per household
    "Latitude": 34.0,   # Los Angeles area
    "Longitude": -118.0
}])

print("\nInput Features:")
print(sample_input.to_string(index=False))

# Make prediction
prediction = model.predict(sample_input)[0]

print(f"\nRaw Prediction: {prediction:.4f} (in $100,000 units)")
print(f"Estimated House Price: ${prediction * 100000:,.2f}")

# Show how prediction is calculated
print("\n" + "=" * 60)
print("PREDICTION CALCULATION BREAKDOWN")
print("=" * 60)
print(f"Base (intercept): {model.intercept_:.4f}")
print("\nContributions from each feature:")
total = model.intercept_
for name, coef, val in zip(feature_names, model.coef_, sample_input.values[0]):
    contribution = coef * val
    total += contribution
    print(f"  {name:15s} ({val:7.1f}) × {coef:+.4f} = {contribution:+.4f}")

print(f"\nTotal: {total:.4f} (matches prediction: {prediction:.4f})")
