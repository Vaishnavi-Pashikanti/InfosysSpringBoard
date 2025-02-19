from pycaret.classification import load_model, predict_model
import pandas as pd

# Load the model
model = load_model(r'C:\Users\vaish\OneDrive\Documents\InfosysSpringBoard\final_xgboost_model.pkl')

# Print the model's feature names
print("Model features:", model.feature_names_in_)

sample_data = {
    'Gender': ['Male'],
    'Age': [21],
    'Height': [1.75],
    'Weight': [80],
    'family_history_with_overweight': ['yes'],
    'frequent_consumption_of_high_caloric_food': ['no'],
    'frequency_of_consumption_of_vegetables': ['frequently'],
    'number_of_main_meals': [3],
    'consumption_of_food_between_meals': ['no'],
    'SMOKE': ['no'],
    'CH2O': [2],
    'calories_consumption_monitoring': ['yes'],
    'physical_activity_frequency': ['sometimes'],
    'time_using_technology_devices': [2],
    'consumption_of_alcohol': ['no'],
    'transportation_used': ['Public_Transportation']
}

# Convert the sample data to a DataFrame
sample_df = pd.DataFrame(sample_data)

# Print the sample data columns
print("Sample data columns:", sample_df.columns.tolist())

# Ensure sample_df has all the required features
for feature in model.feature_names_in_:
    if feature not in sample_df.columns:
        sample_df[feature] = 0  # or any appropriate default value

# Make predictions
predictions = predict_model(model, data=sample_df)

print(predictions)