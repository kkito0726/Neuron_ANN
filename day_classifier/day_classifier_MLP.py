from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Activation, Flatten
from tensorflow.keras.utils import plot_model, to_categorical
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from scipy.optimize import curve_fit

def model_mlp(activation="relu", output_neuron=2):
    model = Sequential([
        # 第1層 (入力層)
        Flatten(),
        Dense(1024, activation=activation),

        # 第2層 (中間層)
        Dense(512, activation=activation),

        # 第3層 (中間層)
        Dense(256, activation=activation),

        # 第4層 (中間層)
        Dense(128, activation=activation),

        # 出力層
        Flatten(),
        Dense(1024, activation=activation),
        Dense(output_neuron, activation="softmax")
    ])

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])

    return model

def nonlinear_fit(x,a,b):
    y = a * np.exp(x / (b+x))
    return y

def nonlinear_fit_loss(x,a,b):
    y = a*np.exp(-x/b)
    return y

def plot_result(history, epochs=20):
    param, cov = curve_fit(nonlinear_fit, np.array(range(epochs))+1, history.history["accuracy"])
    x = np.linspace(0,20,100)
    y = nonlinear_fit(x,*param)
    plt.figure(figsize=(8,6))
    plt.plot(np.array(range(epochs))+1, history.history["accuracy"],'ko', label=("train data"))
    plt.plot(x, y,'k')
    plt.plot(np.array(range(epochs))+1, history.history["val_accuracy"],marker='o', label=("validation data"))
    plt.xlabel("Epochs",fontsize=16)
    plt.ylabel("Accuracy",fontsize=16)
    plt.xticks(range(1,21,1))
    plt.legend()
    plt.show()
    
    param_loss, cov_loss = curve_fit(nonlinear_fit_loss, np.array(range(epochs))+1, history.history["loss"])
    loss_x = np.linspace(0,20,100)
    loss_y = nonlinear_fit_loss(loss_x,*param_loss)
    plt.figure(figsize=(8,6))
    plt.plot(np.array(range(epochs))+1, history.history["loss"],'ko', label=("train data"))
    plt.plot(loss_x, loss_y,'k')
    plt.plot(np.array(range(epochs))+1, history.history["val_loss"],marker='o',label=('validation data'))
    plt.xlabel("Epochs",fontsize=16)
    plt.ylabel("loss",fontsize=16)
    plt.xticks(range(1,21,1))
    plt.legend()
    plt.show()

def main():
    X = np.load("./train_data.npy")
    t = np.load("./label_data.npy")

    model = model_mlp(output_neuron=4)
    model.summary()
    history = model.fit(X, t, epochs=20, validation_split=0.2)

    plot_result(history=history)
    


if __name__ == "__main__":
    main()