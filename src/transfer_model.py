import tensorflow as tf
from tensorflow.keras import layers, models

LAST_BLOCK_START = "block_16_expand"


def build_transfer_model(input_shape=(224, 224, 3), dropout=0.3):
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="head_pool")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="head_output")(x)

    model = models.Model(inputs, outputs, name="mobilenetv2_transfer")
    return model, base


def unfreeze_last_block(base, first_layer=LAST_BLOCK_START):
    base.trainable = True
    unfreeze = False
    for layer in base.layers:
        if layer.name == first_layer:
            unfreeze = True
        layer.trainable = unfreeze


if __name__ == "__main__":
    model, base = build_transfer_model()
    model.summary()