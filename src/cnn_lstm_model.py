import tensorflow as tf
from .cnn_model import build_cnn

def build_cnn_lstm(seq_len=16, img_size=128, channels=1, num_classes=4):
    backbone = build_cnn((img_size, img_size, channels), num_classes=None)  # feature extractor
    inputs = tf.keras.Input(shape=(seq_len, img_size, img_size, channels))
    x = tf.keras.layers.TimeDistributed(backbone)(inputs)
    x = tf.keras.layers.LSTM(256)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="GestureVisionCNNLSTM")
