from keras import layers
from keras.models import Model

from mltu.tensorflow.model_utils import residual_block

def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):
    inputs = layers.Input(shape=input_dim, name="input")

    # Normalize input
    normalized = layers.Lambda(lambda x: x / 255)(inputs)

    # Residual Block 1
    x1 = residual_block(normalized, 16, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x1 = residual_block(x1, 16, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    # Residual Block 2
    x2 = residual_block(x1, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x2 = residual_block(x2, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    # Residual Block 3
    x3 = residual_block(x2, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x3 = residual_block(x3, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    # Residual Block 4 (Added)
    x4 = residual_block(x3, 128, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x4 = residual_block(x4, 128, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    # Residual Block 5 (Added)
    x5 = residual_block(x4, 256, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x5 = residual_block(x5, 256, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    # Reshape for RNN
    squeezed = layers.Reshape((x5.shape[-3] * x5.shape[-2], x5.shape[-1]))(x5)

    # RNN Block
    blstm1 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(squeezed)
    blstm1 = layers.Dropout(dropout)(blstm1)

    blstm2 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(blstm1)
    blstm2 = layers.Dropout(dropout)(blstm2)

    # Dense Output
    output = layers.Dense(output_dim + 1, activation="softmax", name="output")(blstm2)

    # Model
    model = Model(inputs=inputs, outputs=output)
    return model
