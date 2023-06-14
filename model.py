from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import (
    Input,
    AveragePooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def get_model(
        num_classes: int = 4,
        learning_rate: float = 1e-4,
        input_size: int = (224, 224, 3),
        hidden_layer_1: int = 128,
        hidden_layer_2: int = 64,
        dropout: float = 0.2,
        seed: int = 32
):
    print("Preparing model...")

    baseModel = VGG16(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=input_size)
    )

    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(hidden_layer_1, activation="relu")(headModel)
    headModel = Dropout(dropout, seed=seed)(headModel)
    headModel = Dense(hidden_layer_2, activation="relu")(headModel)
    headModel = Dropout(dropout, seed=seed)(headModel)
    headModel = Dense(num_classes, activation='sigmoid')(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)

    model.compile(
        loss=CategoricalCrossentropy(),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    return model
