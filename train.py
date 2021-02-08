#importing the libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os 
import cv2
import random 
import glob
import shutil
import matplotlib.pyplot as plt


os.chdir('./images/images')
if os.path.isdir('train/mask') is False:
    os.makedirs('train/without_mask')
    os.makedirs('train/mask')
    os.makedirs('valid/mask')
    os.makedirs('valid/without_mask')
    os.makedirs('test/mask')
    os.makedirs('test/without_mask')

"""
os.chdir('C:\projects\dataset_trial\images\images\with_mask')

for c in random.sample(glob.glob('*.jpg'),100):
    shutil.move(c,'../valid/mask')
for c in random.sample(glob.glob('*.jpg'),50):
    shutil.move(c,'../test/mask')

# for c in random.sample(glob.glob('*.jpg'),500):
#     shutil.move(c,'../train/mask')


os.chdir('C:\projects\dataset_trial\images\images\without_mask') 
for c in random.sample(glob.glob('*.jpg'),500):
    shutil.move(c,'../train/without_mask')

for c in random.sample(glob.glob('*.jpg'),100):
    shutil.move(c,'../valid/without_mask')

for c in random.sample(glob.glob('*.jpg'),50):
    shutil.move(c,'../test/without_mask')


"""

os.chdir('../../')

train_path = './images/images/train'
valid_path = './images/images/valid'
test_path = './images/images/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['mask','without_mask'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['mask','without_mask'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['mask','without_mask'], batch_size=10, shuffle=False)

imgs, labels = next(train_batches)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
# plotImages(imgs)
# print(labels)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])

model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs=10,
    verbose=2
)


test_imgs, test_labels = next(test_batches)
#plotImages(test_imgs)
#print(test_labels)

predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
np.round(predictions)
test_batches.classes
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

print(test_batches.class_indices)
cm_plot_labels = ['cat','dog']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

model.save('models/mask_detector.h5')


