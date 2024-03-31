import os
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_model(inputs, dataset_train, dataset_val, epochs, best_hps):
    input_layer = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
    lstm_1 = keras.layers.LSTM(best_hps.get('units'), dropout=best_hps.get('dropout'), return_sequences=True)(input_layer)
    lstm_out = keras.layers.LSTM(best_hps.get('units'), dropout=best_hps.get('dropout'), return_sequences=False)(
        lstm_1)
    outputs = keras.layers.Dense(1)(lstm_out)

    model = keras.Model(name="Weather_forecaster", inputs=input_layer, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate')), loss=keras.losses.MeanSquaredError())
    model.summary()

    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val
    )

    # Save the model
    os.makedirs("Model", exist_ok=True)
    model_save_path = "./Model/model.h5"
    model.save(model_save_path)
    print(f"Best model saved to: {model_save_path}")

    loss = history.history["loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    os.makedirs("Model History", exist_ok=True)
    plt.savefig("./Model History/model_history.png")
    plt.show()
    return model, history


# To load the saved model
def load_model(model_path):
    loaded_model = keras.models.load_model(model_path)
    return loaded_model


def generate_data(df, num_rows=48):
    new_rows = []
    for _ in range(num_rows):
        new_row = {}
        for column in df.select_dtypes(include=['number']).columns:
            # Get mean and standard deviation
            mean = df[column].mean()
            std = df[column].std()

            # Generate random data with gaussian distribution.
            synthetic_value = np.random.normal(mean, std)
            new_row[column] = synthetic_value
        new_rows.append(new_row)
    new_df = pd.DataFrame(new_rows)
    return new_df
