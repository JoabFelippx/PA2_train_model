import os

import tensorflow as tf
import cv2 as cv
import numpy as np

from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array

from random import shuffle
import matplotlib.pyplot as plt


# size of the image
TGSIZE = (224, 224)
input_shape = (224, 224, 3)

dataset_path = 'dataset' # path to the dataset folder 


train_dir = f'{dataset_path}/train'
valid_dir = f'{dataset_path}/valid'
test_dir = f'{dataset_path}/test'

train_junior_dir = os.path.join(train_dir, 'junior')
train_gustavo_dir = os.path.join(train_dir, 'gustavo')

valid_junior_dir = os.path.join(valid_dir, 'junior')
valid_gustavo_dir = os.path.join(valid_dir, 'gustavo')

test_junior_dir = os.path.join(test_dir, 'junior')
test_gustavo_dir = os.path.join(test_dir, 'gustavo')


# train and validation generator with data augmentation

def train_valid_generator(train_dir, valid_dir):

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, horizontal_flip=True, fill_mode='nearest')
    valid_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=TGSIZE, batch_size=20, class_mode='binary')
    valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=TGSIZE, batch_size=20, class_mode='binary')
    
    return train_generator, valid_generator


train_generator, valid_generator = train_valid_generator(train_dir, valid_dir)

# load the VGG16 model 
def create_pretrained_model():

    pre_trained_model_vgg16 = VGG16(input_shape=input_shape, weights=None, include_top=False)
    
    for layer in pre_trained_model_vgg16.layers:
        layer.trainable = False
    
    return pre_trained_model_vgg16


pre_trained_model_vgg16 = create_pretrained_model()

# show the model summary
pre_trained_model_vgg16.summary()


# callback to stop the training when the accuracy reaches 99.9%
class myCallback(tf.keras.callbacks.Callback):
  
  def on_epoch_end(self, epoch, logs={}):

    if(logs.get('accuracy')>0.999):
      print("\nAtingi 99,9% de precisão, portanto, estou cancelando o treinamento!!")
      self.model.stop_training = True


# output of the last layer of the VGG16 model
def output_last_layer(pre_trained_model_vgg16):

    last_layer = pre_trained_model_vgg16.get_layer('block5_pool')
    last_output = last_layer.output
    
    return last_output

last_output = output_last_layer(pre_trained_model_vgg16)


# create the final model 
def create_final_model(pre_trained_model_vgg16, last_output):

    x = tf.keras.layers.Flatten()(last_output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)    
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    model = tf.keras.Model(pre_trained_model_vgg16.input, x)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

model = create_final_model(pre_trained_model_vgg16, last_output)


model.load_weights('model_weight.h5')


test_images = [f'{test_junior_dir}/{img}' for img in os.listdir(test_junior_dir)] + [f'{test_gustavo_dir}/{img}' for img in os.listdir(test_gustavo_dir)]

shuffle(test_images)

for img_path in test_images:
    loaded_img = load_img(img_path, target_size=TGSIZE)

    img_array = img_to_array(loaded_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    model_prediction = model.predict(img_array)
    print(model_prediction)

    if model_prediction[0][0] > 0.5:
        print(f'Gustavo: {model_prediction[0][0]}\timage: {img_path}')
    elif model_prediction[0][1] > 0.5:
        print(f'Junior: {model_prediction[0][1]}\timage: {img_path}')

    # if runing code outside of jupyter notebook, uncomment the following lines
    # cv.imshow('image', cv.imread(img_path))
    # cv.waitKey(0)
    # cv.destroyAllWindows()
