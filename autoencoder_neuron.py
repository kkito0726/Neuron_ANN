import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(50, 64, 1)),
            layers.Conv2D(16, (3, 3), activation="relu", padding="same", strides=2),
            layers.Conv2D( 8, (3, 3), activation="relu", padding="same", strides=2),
            layers.Conv2D( 4, (3, 3), activation="relu", padding="same", strides=2),
            layers.Flatten(),
            layers.Dense((7*7))
        ])

        self.decoder = tf.keras.Sequential([
            layers.Reshape((7, 7, 1)),
            layers.Conv2DTranspose( 4, (3, 3), activation="relu", padding="same", strides=2),
            layers.Conv2DTranspose( 8, kernel_size=3, strides=2, activation="relu", padding="same"),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation="relu", padding="same"),
            layers.Conv2D(1, kernel_size=(3, 3), activation="sigmoid", padding="same"),
            layers.Flatten(),
            layers.Dense(50 * 64),
            layers.Reshape((50, 64))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
autoencoder = Autoencoder()
autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())



if __name__ == "__main__":
    print("\n")
    # numpy配列を読み込む
    file_path = input("50*64のテンソルデータのパスを入力 (numpyのバイナリーファイル.npy): ")
    data = np.load(file_path)

    # 訓練データとテストデータに分ける
    x_train, x_test = train_test_split(data, test_size=0.2, random_state=0)
    print(x_train.shape, x_test.shape)

    # 定義したモデルをもとに学習
    autoencoder.fit(
        x_train, x_train,
        epochs=10,
        shuffle=True,
        validation_data=(x_test, x_test)
    )
    print(autoencoder.encoder.summary(), autoencoder.decoder.summary())


    encoded_imgs = autoencoder(x_test).numpy()
    decoded_imgs = autoencoder(encoded_imgs).numpy()

    n = 10
    plt.figure(figsize=(10, 5))
    for i in range(n):
        # オリジナル画像を表示
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i])
        if i == 0: plt.title("original") 
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # 再構成画像を表示
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        if i == 0: plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()