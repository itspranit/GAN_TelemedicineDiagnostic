import tensorflow as tf
from config import *
from dataset import build_dataset
from networks.generator import build_generator
from networks.discriminator import build_discriminator
from losses import perceptual_loss
from utils import save_images

generator=build_generator(lr_shape=(LR_SIZE,LR_SIZE,CHANNELS))
discriminator=build_discriminator(hr_shape=(HR_SIZE,HR_SIZE,CHANNELS))
bce=tf.keras.losses.BinaryCrossentropy(from_logits=False)
gen_opt=tf.keras.optimizers.Adam(LEARNING_RATE)
disc_opt=tf.keras.optimizers.Adam(LEARNING_RATE)

@tf.function
def train_step(lr,hr):
    with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:
        fake_hr=generator(lr,training=True)
        real_out=discriminator(hr,training=True)
        fake_out=discriminator(fake_hr,training=True)
        
        disc_loss_real=bce(tf.ones_like(real_out),real_out)
        disc_loss_fake=bce(tf.zeros_like(fake_out),fake_out)
        disc_loss=disc_loss_real+disc_loss_fake
        
        adv_loss=bce(tf.ones_like(fake_out),fake_out)
        p_loss=perceptual_loss(hr,fake_hr)
        gen_loss=(1e-3*adv_loss)+p_loss
        
    gen_grads=gen_tape.gradient(gen_loss,generator.trainable_variables)
    disc_grads=disc_tape.gradient(disc_loss,discriminator.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grads,generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grads,discriminator.trainable_variables))
    return gen_loss,disc_loss,fake_hr

def train():
    dataset=build_dataset(DATASET_DIR)
    for epoch in range(EPOCHS):
        for lr,hr in dataset:
            g_loss,d_loss,fake_hr=train_step(lr,hr)
            
        print(f"Epoch {epoch+1}/{EPOCHS} | D:{d_loss:.4f} | G:{g_loss:.4f}")
        
        if (epoch+1)%10==0:
            generator.save_weights(f"{WEIGHTS_DIR}/gen_{epoch+1}.weights.h5")
            save_images(epoch+1,lr,hr,fake_hr,LOGS_DIR)

if __name__=="__main__":
    train()