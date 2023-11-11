import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Sample data
data = [('cosa', 2), ('cosa', 3)]

# Convert list of tuples to DataFrame
df = pd.DataFrame(data, columns=['Category', 'Count'])

# Create an instance of OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Fit and transform the DataFrame
encoded_array = encoder.fit_transform(df[['Category', 'Count']])

# Print the result
print(encoded_array)