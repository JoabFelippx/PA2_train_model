import os

import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt


# size of the image
TGSIZE = (224, 224)
input_shape = (224, 224, 3)

dataset_path = '' # path to the dataset folder 


train_dir = f'{dataset_path}/train'
valid_dir = f'{dataset_path}/dataset/valid'

train_junior_dir = os.path.join(train_dir, 'junior')
train_gustavo_dir = os.path.join(train_dir, 'gustavo')

valid_junior_dir = os.path.join(valid_dir, 'junior')
valid_gustavo_dir = os.path.join(valid_dir, 'gustavo')


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

callbacks = myCallback()
history = model.fit(train_generator, validation_data=valid_generator, epochs=100, verbose=2, callbacks=[callbacks])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Acurácia de Treino')
plt.plot(epochs, val_acc, 'b', label='Acurácia de Validação')
plt.title('Acurácia de Treino e Validação')
plt.legend(loc=0)
plt.figure()

plt.show()
