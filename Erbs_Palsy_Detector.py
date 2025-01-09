import pandas as pd
import pickle

model_filename = 'logistic_regression_model.pkl'
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)
df = pd.read_csv('erbs_palsy_artificial_data.csv')

example_row = df.iloc[[0]]

example_input = example_row.drop(columns=['erbs_palsy'])

example_input = pd.get_dummies(example_input, drop_first=True)

expected_columns = loaded_model.coef_.shape[1]
example_input = example_input.reindex(columns=loaded_model.feature_names_in_, fill_value=0)

prediction = loaded_model.predict(example_input)

if prediction[0] == 1:
    result = "There is a higher chance of Erb's palsy."
else:
    result = "There is a lower chance of Erb's palsy."

print("Test Input:")
print(example_input)
print(result)
