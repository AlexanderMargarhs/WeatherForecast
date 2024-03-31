import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def get_feature_importance(weather_data):
	# Number of classes
	num_classes = 1

	weather_data = weather_data.drop(columns=['Summary', 'Daily Summary'])
	# Separate features (X) and target variable (y)
	X = weather_data.drop(columns=['Apparent Temperature (C)'])  # Features
	y = weather_data['Temperature (C)']  # Target variable

	# Split the data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# 1. Feature Importance from Tree-based Models
	# Train a Random Forest classifier to obtain feature importance
	rf = RandomForestRegressor(n_estimators=100, random_state=42)
	rf.fit(X_train, y_train)

	# Get feature importance scores
	feature_importance = rf.feature_importances_
	feature_names = X.columns

	# Select features with importance above a certain threshold
	threshold = 0.05  # Example threshold
	selected_features_rf = feature_names[feature_importance > threshold]

	# 2. Recursive Feature Elimination (RFE)
	# Initialize RFE with a RandomForestRegressor
	rfe = RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=5)
	rfe.fit(X_train, y_train)

	# Get selected features from RFE
	selected_features_rfe = feature_names[rfe.support_]

	# Print selected features
	print("Selected Features from Random Forest Feature Importance:")
	print(selected_features_rf)
	print("\nSelected Features from Recursive Feature Elimination (RFE):")
	print(selected_features_rfe)

	# Drop columns not in the list
	columns_to_drop = [col for col in X.columns if col not in selected_features_rf]
	rf_data = X.drop(columns=columns_to_drop)

	# Drop columns not in the list
	columns_to_drop = [col for col in X.columns if col not in selected_features_rfe]
	rfe_data = X.drop(columns=columns_to_drop)

	return num_classes, rf_data, rfe_data


def normalize(data):
	data_mean = data.mean(axis=0)
	data_std = data.std(axis=0)
	return (data - data_mean) / data_std


def get_features(weather_data, features):
	feature_data = pd.DataFrame()
	for feature in features:
		print(feature)
		feature_data = pd.concat([feature_data, weather_data[feature]], axis=1)
	return feature_data
