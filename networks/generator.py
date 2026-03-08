# networks/generator.py
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, PReLU, Add, UpSampling2D, Input
from tensorflow.keras.models import Model

def residual_block(x):
    """residual block to learn complex features."""
    res=Conv2D(64,kernel_size=3,padding='same')(x)

    res=BatchNormalization(momentum=0.8)(res)
    res=PReLU(shared_axes=[1, 2])(res)
    res=Conv2D(64, kernel_size=3, padding='same')(res)
    res=BatchNormalization(momentum=0.8)(res)

    return Add()([x, res])

def upsample_block(x):
    """increases the dimensions of the image by factor of 2."""
    x=Conv2D(256, kernel_size=3, padding='same')(x)
    x=UpSampling2D(size=2)(x)
    x=PReLU(shared_axes=[1, 2])(x)
    return x

def build_generator(lr_shape=(64, 64, 3)):
    """assembles the full Super-Resolution Generator."""
    inputs =Input(shape=lr_shape)
    
    conv1=Conv2D(64, kernel_size=9, padding='same')(inputs)
    conv1=PReLU(shared_axes=[1, 2])(conv1)

    res=conv1
    for _ in range(16):
        res=residual_block(res)
        
  
    conv2=Conv2D(64, kernel_size=3, padding='same')(res)
    conv2=BatchNormalization(momentum=0.8)(conv2)
    conv2=Add()([conv2, conv1])
    

    up=upsample_block(conv2)
    up=upsample_block(up)
    
   
    outputs=Conv2D(3, kernel_size=9, padding='same', activation='tanh')(up)
    
    return Model(inputs, outputs, name="Generator")