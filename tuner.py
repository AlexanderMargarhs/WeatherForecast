import tensorflow.keras as keras


def build_model(hp, inputs, sequence_to_predict):
	input_layer = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
	lstm_1 = keras.layers.LSTM(
		units=hp.Int('units', min_value=32, max_value=256, step=32),
		dropout=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1),
		return_sequences=True
	)(input_layer)
	lstm_out = keras.layers.LSTM(
		units=hp.Int('units', min_value=32, max_value=256, step=32),
		dropout=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1),
		return_sequences=False
	)(lstm_1)
	outputs = keras.layers.Dense(1)(lstm_out)

	model = keras.Model(inputs=input_layer, outputs=outputs)

	model.compile(optimizer=keras.optimizers.Adam(
		hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
		loss='mse',
		metrics=['mae'])

	return model
