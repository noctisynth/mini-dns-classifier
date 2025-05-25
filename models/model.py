from typing import List, Tuple
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from keras.layers import Conv1D, Embedding
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.regularizers import l2

import tensorflow as tf


class CharCNNKim:
    def __init__(
        self,
        input_size: int = 256,
        alphabet_size: int = 69,
        embedding_size: int = 128,
        conv_layers: List[Tuple[int, int]] = [(256, 7), (256, 5), (256, 3)],
        fc_layers: List[int] = [1024, 512],
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        self.input_size = input_size
        self.alphabet_size = alphabet_size
        self.embedding_size = embedding_size
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.model = self._build_model()

    def _build_model(self) -> Model:
        inputs = Input(shape=(self.input_size,), name="sent_input", dtype="int64")

        x = Embedding(
            self.alphabet_size + 1,
            self.embedding_size,
            input_length=self.input_size,
            embeddings_regularizer=l2(1e-4),
        )(inputs)

        x = Dropout(0.2)(x)

        conv_outputs = []
        for i, (filters, kernel) in enumerate(self.conv_layers):
            conv = Conv1D(
                filters=filters,
                kernel_size=kernel,
                activation="relu",
                padding="same",
                kernel_regularizer=l2(1e-4),
            )(x)
            conv = BatchNormalization()(conv)
            conv_outputs.extend(
                [GlobalMaxPooling1D()(conv), GlobalAveragePooling1D()(conv)]
            )

        x = Concatenate()(conv_outputs)

        for units in self.fc_layers:
            x = Dense(units, activation="relu", kernel_regularizer=l2(1e-4))(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout)(x)

        outputs = Dense(
            self.num_classes,
            activation="softmax",
            kernel_regularizer=l2(1e-4),
        )(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        return model

    def save(self, path: str):
        self.model.save(path)

    def load_weights(self, path: str) -> None:
        self.model.load_weights(path)
