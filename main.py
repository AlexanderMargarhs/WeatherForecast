import os
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from tuner import build_model
import matplotlib.pyplot as plt
from models import run_model, load_model
from data_cleaning import read_file, visualize_correlation
from sklearn.metrics import mean_absolute_error, mean_squared_error
from feature_importance import get_feature_importance, get_features, normalize

if __name__ == "__main__":
	weather_data, label_encoder = read_file()

	# Set dates as indices
	weather_data.set_index('Formatted Date', inplace=True)

	# Visualise the correlation
	visualize_correlation(weather_data)
	num_classes, selected_features_rf, selected_features_rfe = get_feature_importance(weather_data)
	weather_data = weather_data.drop(columns=['Summary', 'Daily Summary'])

	# Get features and create the new dataset
	features = get_features(weather_data, selected_features_rfe)
	features = normalize(features.values)
	features = pd.DataFrame(features)

	# Get train-validation sets.
	training_size = int(0.8 * features.shape[0])
	train_data = features.loc[0: training_size - 1]
	val_data = features.loc[training_size:]

	# Set series to read each time
	epochs = 100
	days = 3
	timestamps = 24
	days_to_predict = 3
	sequence_to_predict = timestamps * days_to_predict
	start = days * timestamps + sequence_to_predict
	end = start + training_size

	# Split data
	x_train = train_data.values
	y_train = features.iloc[start:end][[0]]

	# Set sequence length
	sequence_length = int((days * timestamps + days_to_predict * timestamps) / (days_to_predict * timestamps))

	# Set dataset as a timeseries
	dataset_train = keras.preprocessing.timeseries_dataset_from_array(
		data=x_train,
		targets=y_train,
		sequence_length=sequence_length,
		sampling_rate=6,
		batch_size=16,
	)

	# Create validation set
	x_val_end = len(val_data) - start

	label_start = training_size + start

	x_val = val_data.iloc[:x_val_end][[i for i in range(len(features.columns))]].values
	y_val = features.iloc[label_start:][[0]]

	dataset_val = keras.preprocessing.timeseries_dataset_from_array(
		data=x_val,
		targets=y_val,
		sequence_length=sequence_length,
		sampling_rate=6,
		batch_size=16,
	)

	inputs = []
	targets = []

	# Iterate through all batches
	for batch in dataset_train:
		batch_inputs, batch_targets = batch
		inputs.append(batch_inputs)
		targets.append(batch_targets)

	# Concatenate batches to get all inputs and targets
	inputs = tf.concat(inputs, axis=0)
	targets = tf.concat(targets, axis=0)

	# Instantiate the tuner
	tuner = kt.Hyperband(
		lambda hp: build_model(hp, inputs, sequence_to_predict),
		objective='val_loss',
		max_epochs=10,
		factor=3,
		directory='weather_forecast',
		project_name='lstm_tuning'
	)

	# Perform the hyperparameter search
	tuner.search(inputs, targets, epochs=50, validation_split=0.2)

	# Get the best hyperparameters
	best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
	model, history_data = run_model(inputs, dataset_train, dataset_val, epochs, best_hps)

	# Loading the saved model
	# loaded_model = load_model("./Model/model.h5")

	# Predict temperatures using the trained model
	predictions = model.predict(dataset_val)

	# Concatenate predictions
	predictions = np.concatenate(predictions, axis=0)
	y_val = y_val[:len(predictions)]

	# Calculate evaluation metrics
	mae = mean_absolute_error(y_val, predictions)
	mse = mean_squared_error(y_val, predictions)
	rmse = np.sqrt(mse)

	print("Mean Absolute Error (MAE):", mae)
	print("Mean Squared Error (MSE):", mse)
	print("Root Mean Squared Error (RMSE):", rmse)

	# Ensure predictions and y_val have the same length
	min_length = min(len(y_val), len(predictions))
	y_val = y_val[:min_length]
	predictions = predictions[:min_length]

	# Adjust the range of indices for plotting based on the minimum length
	plot_range = min(sequence_to_predict, min_length)

	# Plotting predicted and actual temperatures
	plt.figure(figsize=(10, 6))
	plt.plot(val_data.index[-plot_range:], y_val[-plot_range:], label='Actual')
	plt.plot(val_data.index[-plot_range:], predictions[-plot_range:], label='Predicted')
	plt.title('Temperature Prediction vs Actual')
	plt.xlabel('Time')
	plt.ylabel('Temperature')
	plt.legend()
	os.makedirs("Prediction", exist_ok=True)
	plt.savefig("./Prediction/prediction.png")
	plt.show()

