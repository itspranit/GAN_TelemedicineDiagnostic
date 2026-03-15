import tensorflow as tf
from config import *
from tensorflow.keras.applications.vgg19 import VGG19 # type: ignore
from tensorflow.keras.models import Model

def build_vgg():
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=(HR_SIZE, HR_SIZE, CHANNELS))
    vgg.trainable=False
    return Model(inputs=vgg.input,outputs=vgg.get_layer('block5_conv4').output)

vgg_model=build_vgg()

@tf.function
def perceptual_loss(hr_real,hr_fake):
    hr_real_scaled=(hr_real+1.0)*127.5
    hr_fake_scaled=(hr_fake+1.0)*127.5

    hr_real_preprocessed=tf.keras.applications.vgg19.preprocess_input(hr_real_scaled)
    hr_fake_preprocessed=tf.keras.applications.vgg19.preprocess_input(hr_fake_scaled)

    real_features=vgg_model(hr_real_preprocessed)
    fake_features=vgg_model(hr_fake_preprocessed)

    return tf.reduce_mean(tf.square(real_features-fake_features))