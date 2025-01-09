import numpy as np
import pandas as pd

# Number of data points
num_samples = 1000

# Generate random data for factors
np.random.seed(42)

data = {
    'diabetes': np.random.choice([0, 1], size=num_samples, p=[0.85, 0.15]),  # 15% chance of diabetes
    'shoulder_width_cm': np.random.normal(12, 2, num_samples),  # Average shoulder width ~12cm with std dev of 2cm
    'pelvis_inner_width_cm': np.random.normal(13, 1.5, num_samples),  # Average pelvis width ~13cm with std dev of 1.5cm
    'child_alignment_suction_cup': np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1]),  # 10% requiring suction
    'gestational_age_weeks': np.random.normal(39, 1.5, num_samples),  # Gestation average ~39 weeks
    'birth_weight_kg': np.random.normal(3.5, 0.5, num_samples),  # Average birth weight ~3.5kg with std dev 0.5kg
    'maternal_BMI': np.random.normal(26, 5, num_samples),  # Average maternal BMI ~26
    'labor_duration_hours': np.random.normal(12, 3, num_samples),  # Average labor ~12 hours
    'forceps_or_vacuum_delivery': np.random.choice([0, 1], size=num_samples, p=[0.85, 0.15]),  # 15% use of tools
    'delivery_type': np.random.choice(['vaginal', 'cesarean'], size=num_samples, p=[0.75, 0.25]),  # 75% vaginal, 25% cesarean
    'head_circumference_cm': np.random.normal(34, 2, num_samples),  # Average head circumference ~34cm
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Generating the target variable for Erb's palsy (just for demo purposes)
# Assume that Erb's palsy is more likely to occur in cases with larger shoulder width, smaller pelvis width, vacuum-assisted delivery, etc.
df['erbs_palsy'] = (
    (df['shoulder_width_cm'] > 13) & 
    (df['pelvis_inner_width_cm'] < 12) &
    (df['child_alignment_suction_cup'] == 1) |
    (df['forceps_or_vacuum_delivery'] == 1)
).astype(int)

# Preview the data
print(df.head())

# Save the artificial dataset to CSV
df.to_csv('erbs_palsy_artificial_data.csv', index=False)
