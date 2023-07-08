from keras.layers import Conv3D, Conv3DTranspose, Input, concatenate, LeakyReLU
from keras.models import Model
from tensorflow_addons.layers import InstanceNormalization


def binary_double_conv_block(x, n_filters, activation='relu'):
    x = InstanceNormalization()(x)
    x = Conv3D(n_filters, 3, padding="same", activation=activation)(x)
    x2 = InstanceNormalization()(x)
    x2 = Conv3D(n_filters, 3, padding="same", activation=activation)(x2)
    x2 = InstanceNormalization()(x2)
    return x2


def binary_model(img_height: int, img_width: int,
                 img_depth: int, img_channels: int,
                 num_classes: int, channels: int, activation="relu"):
    # Build the model
    inputs = Input((img_height, img_width, img_depth, img_channels))
    s = inputs

    # Contraction pat
    c2 = binary_double_conv_block(s, channels * 2)
    p2 = Conv3D(channels * 2, (3, 3, 3), strides=(2, 2, 2), activation=activation, padding='same')(c2)

    c3 = binary_double_conv_block(p2, channels * 4)
    p3 = Conv3D(channels * 4, (3, 3, 3), strides=(2, 2, 2), activation=activation, padding='same')(c3)

    c4 = binary_double_conv_block(p3, channels * 8)
    p4 = Conv3D(channels * 8, (3, 3, 3), strides=(2, 2, 2), activation=activation, padding='same')(c4)

    c5 = binary_double_conv_block(p4, channels * 10)

    # Expansive path
    u6 = Conv3DTranspose(channels * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = binary_double_conv_block(u6, channels * 8)

    u7 = Conv3DTranspose(channels * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = binary_double_conv_block(u7, channels * 4)

    u8 = Conv3DTranspose(channels * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = binary_double_conv_block(u8, channels * 2)

    outputs = Conv3D(num_classes, (1, 1, 1), activation='sigmoid')(c8)

    return Model(inputs=[inputs], outputs=[outputs])


def double_conv_block(x, n_filters, activation):
    x1 = InstanceNormalization()(x)
    x1 = Conv3D(n_filters, 3, padding="same", activation=activation)(x1)
    x2 = InstanceNormalization()(x1)
    x2 = Conv3D(n_filters, 3, padding="same", activation=activation)(x2)
    return x2


# Change to binary model

def brain_tumor_model(img_height: int, img_width: int,
                      img_depth: int, img_channels: int,
                      num_classes: int, activation=LeakyReLU(alpha=0.05), channels: int = 32):
    inputs = Input((img_height, img_width, img_depth, img_channels))

    c1 = double_conv_block(inputs, channels * 2, activation)
    p1 = Conv3D(channels * 2, (3, 3, 3), strides=(2, 2, 2), activation=activation, padding='same')(c1)

    c2 = double_conv_block(p1, channels * 4, activation)
    p2 = Conv3D(channels * 4, (3, 3, 3), strides=(2, 2, 2), activation=activation, padding='same')(c2)

    c3 = double_conv_block(p2, channels * 8, activation)
    p3 = Conv3D(channels * 8, (3, 3, 3), strides=(2, 2, 2), activation=activation, padding='same')(c3)

    c4 = double_conv_block(p3, channels * 10, activation)

    u1 = Conv3DTranspose(channels * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(c4)
    u1 = concatenate([u1, c3])
    c5 = double_conv_block(u1, channels * 8, activation)

    u2 = Conv3DTranspose(channels * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u2 = concatenate([u2, c2])
    c6 = double_conv_block(u2, channels * 4, activation)

    u3 = Conv3DTranspose(channels * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u3 = concatenate([u3, c1])
    c7 = double_conv_block(u3, channels * 2, activation)

    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c7)

    return Model(inputs=[inputs], outputs=[outputs])
