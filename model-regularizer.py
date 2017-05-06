import csv
import cv2
import numpy as np 
import sklearn
from sklearn.utils import shuffle

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
# Validation Split
train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)  

# Steering angle adjustment for left and right images
steering_correction = 0.25
def preprocess(im):
    # resize 
    im = cv2.resize(im,(320,160)) 
    # convert to YUV
    im = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    return im

def generator(samples, batch_size = 16):
    num_samples = len(samples)
    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #For windows simulator use split('\\')
                nameleft, namecenter, nameright = './data/IMG/'+batch_sample[1].split('/')[-1], './data/IMG/'+batch_sample[0].split('/')[-1], './data/IMG/'+batch_sample[2].split('/')[-1]
#                               nameleft, namecenter, nameright = './data/IMG/'+batch_sample[1].split('\\')[-1], './data/IMG/'+batch_sample[0].split('\\')[-1], './data/IMG/'+batch_sample[2].split('\\')[-1]
                left_image, center_image, right_image = preprocess(cv2.imread(nameleft)), preprocess(cv2.imread(namecenter)), preprocess(cv2.imread(nameright))
                center_angle = float(batch_sample[3])
                # For each sample, we generate 3 image sets: left, right, and center.
                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                if abs(center_angle) < 0.25:
                    left_angle, right_angle = center_angle, center_angle
                else:
                    left_angle, right_angle = center_angle + steering_correction, center_angle - steering_correction
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)
                
            # Augment images with flipped set
            aug_images, aug_angles = [], []
            for image, angle in zip(images,angles):
                aug_images.append(image)
                aug_angles.append(angle)
                aug_images.append(cv2.flip(image,1))
                aug_angles.append( -1.0 * angle)
                
            X_train = np.array(aug_images)
            y_train = np.array(aug_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Batch size returned will be  x6 due to adding left and right images, then adding flipped ones.  
train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

# Configure Keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras_tqdm import TQDMNotebookCallback # Better Keras Progressbar
from keras import backend as K
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
import gc

# Need to limit GPU memory to 3.5GB given our lovely GTX 970 memory issue.
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = .85
set_session(tf.Session(config=config))


# Train and save model
ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2),  W_regularizer=l2(0.001)))
model.add(ELU())
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(36,5,5, subsample=(2,2),  W_regularizer=l2(0.001)))
model.add(ELU())
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(48,5,5, subsample=(2,2),  W_regularizer=l2(0.001)))
model.add(ELU())
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(64,3,3, subsample=(1,1),  W_regularizer=l2(0.001)))
model.add(ELU())
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(64,3,3, subsample=(1,1),  W_regularizer=l2(0.001)))
model.add(ELU())
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Flatten())
model.add(ELU())
model.add(Dense(100,  W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dense(50,   W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dense(10,   W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam(lr = 0.001) )
model.fit_generator(train_generator, samples_per_epoch= 6 * len(train_samples), 
                    validation_data=validation_generator, nb_val_samples=len(validation_samples), 
                    verbose=1, nb_epoch=5)
model.save('model5.h5')