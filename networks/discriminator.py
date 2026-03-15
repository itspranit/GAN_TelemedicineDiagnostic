import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,LeakyReLU,Dense,Flatten,Input
from tensorflow.keras.models import Model

def discriminator_block(x,filters,strides=1,batch_norm=True):
    x=Conv2D(filters,kernel_size=3,strides=strides,padding='same')(x)
    if batch_norm:
        x=BatchNormalization(momentum=0.8)(x)
    x=LeakyReLU(negative_slope=0.2)(x)
    return x

def build_discriminator(hr_shape=(256,256,3)):
    inputs=Input(shape=hr_shape)

    x=discriminator_block(inputs,64,batch_norm=False)
    x=discriminator_block(x,64,strides=2)
    x=discriminator_block(x,128)
    x=discriminator_block(x,128,strides=2)
    x=discriminator_block(x,256)
    x=discriminator_block(x,256,strides=2)
    x=discriminator_block(x,512)
    x=discriminator_block(x,512,strides=2)

    x=Flatten()(x)
    x=Dense(1024)(x)
    x=LeakyReLU(negative_slope=0.2)(x)
    outputs=Dense(1,activation='sigmoid')(x)

    return Model(inputs,outputs,name="Discriminator")