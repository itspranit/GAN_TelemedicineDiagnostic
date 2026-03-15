import tensorflow as tf
import os
from config import *

def load_and_process_image(image_path):
    image=tf.io.read_file(image_path)
    image=tf.image.decode_jpeg(image,channels=3)
    hr_image=tf.image.resize(image,[HR_SIZE,HR_SIZE])
    hr_image=(hr_image/127.5)-1.0
    lr_image=tf.image.resize(image,[LR_SIZE,LR_SIZE],method=tf.image.ResizeMethod.BICUBIC)
    lr_image=lr_image/255.0
    return lr_image,hr_image

def build_dataset(dataset_dir):
    file_pattern=os.path.join(dataset_dir,'*.jpg')
    dataset=tf.data.Dataset.list_files(file_pattern)
    dataset=dataset.map(load_and_process_image,num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset=dataset.cache() 
    
    dataset=dataset.shuffle(buffer_size=1000)
    dataset=dataset.batch(BATCH_SIZE)
    dataset=dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset