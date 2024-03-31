import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder


def read_file():
	# Load the dataset.
	weather_data = pd.read_csv('./Dataset/interview_dataset.csv')

	# Exploratory Data Analysis.
	print(weather_data.head())  # View the first few rows of the dataset.
	print(weather_data.info())  # Check data types and missing values.
	print(weather_data.describe())  # Summary statistics.

	# Check for missing values.
	missing_values = weather_data.isnull().sum()
	print("Missing Values:", missing_values)

	# Check and remove outliers.
	weather_data = remove_outliers(weather_data)

	# Drop rows where the imputer didn't have sufficient data.
	weather_data = weather_data.dropna()

	# Replace the original column with the imputed values
	weather_data['Precip Type'], label_encoder = data_imputation(weather_data)

	# Check for missing values.
	missing_values = weather_data.isnull().sum()
	print("Missing Values After Clean Up:", missing_values)

	# Save the DataFrame to a CSV file
	weather_data.to_csv('./Dataset/updated_dataset.csv', index=False)

	return weather_data, label_encoder


def data_imputation(weather_data):
	# Check the column names in the DataFrame
	print(weather_data.columns)

	# Assuming the column name is 'Precip Type', use the correct column name from your DataFrame
	column_with_missing = weather_data['Precip Type']  # Ensure 'Precip Type' is the correct column name

	# Label encode the categorical column
	label_encoder = LabelEncoder()
	column_with_missing_encoded = label_encoder.fit_transform(
		column_with_missing.astype(str))  # Ensure column values are converted to strings

	# Convert the encoded column to a DataFrame
	df = pd.DataFrame(column_with_missing_encoded, columns=['Precip Type'])

	# Reshape the column to 2D array
	column_with_missing_2d = df['Precip Type'].values.reshape(-1, 1)

	# Initialize the KNNImputer with k neighbors
	imputer = KNNImputer(n_neighbors=2, weights="uniform")

	# Impute missing values using KNN imputation
	imputed_column_2d = imputer.fit_transform(column_with_missing_2d)

	# Flatten the imputed column back to 1D array
	imputed_column = imputed_column_2d.flatten()

	return imputed_column, label_encoder


def remove_outliers(df, threshold=1.5):
	numeric_columns = df.select_dtypes(include=['number']).columns

	for column in numeric_columns:
		Q1 = df[column].quantile(0.25)
		Q3 = df[column].quantile(0.75)
		IQR = Q3 - Q1
		lower_bound = Q1 - threshold * IQR
		upper_bound = Q3 + threshold * IQR
		df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

	return df


def visualize_correlation(weather_data):
	# Check if the directory Correlation exists, if it doesn't create it
	os.makedirs("Correlation", exist_ok=True)
	# Visualize the distribution of numerical features
	sns.pairplot(weather_data, diag_kind='kde')
	# Save image
	plt.savefig("./Correlation/distributions.png")
	plt.show()

	# Correlation analysis
	correlation_matrix = weather_data.corr(numeric_only=True)
	plt.figure(figsize=(12, 8))
	sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
	plt.title('Correlation Matrix')
	# Save image
	plt.savefig("./Correlation/correlation_matrix.png")
	plt.show()


