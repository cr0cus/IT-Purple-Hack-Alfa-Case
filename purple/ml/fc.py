import tensorflow as tf
from keras import layers
from keras.models import Model


class FCClassifier(Model):
    def __init__(self, input_shape: int):
        super(FCClassifier, self).__init__()
        self.model = tf.keras.Sequential([
            layers.Input(shape=(input_shape,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(17, activation='softmax')
        ])

    def call(self, x):
        label = self.model(x)
        return label


if __name__ == "__main__":
    model = FCClassifier(input_shape=223)
    model.model.summary()

