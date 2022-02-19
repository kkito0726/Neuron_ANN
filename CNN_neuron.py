import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.utils import plot_model, to_categorical
import matplotlib.pyplot as plt
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split

def model_cnn(activation="relu", output_neuron=2):
    model = Sequential([
        # 第1層 (入力層)
        Conv2D(32, (3, 3), input_shape=(50,64,1), activation=activation),

        # 第2層 (中間層)
        Conv2D(32, (3, 3), activation=activation),
        MaxPool2D(pool_size=(2, 2)),

        # 第3層 (中間層)
        Conv2D(32, (3, 3), activation=activation),
        MaxPool2D(pool_size=(2, 2)),

        # 第4層 (中間層)
        Conv2D(32, (3, 3), activation=activation),
        MaxPool2D(pool_size=(2, 2)),

        # 出力層
        Flatten(),
        Dense(1024, activation=activation),
        Dense(output_neuron, activation="softmax")
    ])

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])

    return model

if __name__ == "__main__":
    model = model_cnn()
    model.summary()